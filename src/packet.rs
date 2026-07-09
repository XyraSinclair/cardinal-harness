//! The judgment packet: belief as a medium.
//!
//! A packet is a sealed, content-addressed bundle of pairwise evidence
//! about one attribute over a set of entities: who judged, under what
//! instrument, and every observation as a (log-ratio, precision) pair.
//! Two parties who have never coordinated can elicit independently,
//! exchange packets, and FUSE them — and the fused posterior is
//! **byte-identical** to what a single party holding all the evidence
//! would compute.
//!
//! Why byte-identical and not merely close: the solver depends only on
//! the observation MULTISET — the free commutative monoid; under Huber
//! weights no smaller mergeable sufficient statistic exists, which is
//! why packets carry observations, not moments (pinned in
//! tests/program_equivalence.rs) — and `fuse` canonicalizes that multiset (sorts
//! entities by id, observations by content) before the solve, making the
//! float operations themselves order-identical. What was a 1e-9
//! tolerance under arbitrary arrival order becomes `==` under the
//! protocol.
//!
//! The CRDT statement, made precise (the loose form — "a CRDT of
//! belief" over raw fuse — was killed by the 2026-07-09 red team, which
//! caught that fusion without dedup double-counts a re-delivered
//! packet): the replicated STATE is the SET of packets keyed by content
//! address. `fuse` dedups by packet id, so merge is set union —
//! commutative, associative, AND idempotent — and the posterior is a
//! pure function of the state (a G-Set CRDT). What id-dedup cannot see:
//! two DISTINCT packets built from overlapping raw draws still
//! double-count; evidence-disjointness of distinct packets is the
//! provenance layer's contract (#42/#46), not this function's.
//!
//! Integrity: the packet id is blake3 over a domain-tagged CANONICAL
//! BYTE encoding (length-prefixed strings, f64 bit patterns — no JSON
//! float round-trip in the identity path). Entity texts travel as
//! hashes: light packets that still pin exactly what was judged; fusion
//! refuses when two packets disagree about an entity id's content hash.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use seriate::ontology::ContentId;

use crate::rating_engine::{AttributeParams, Config, Observation, RaterParams, RatingEngine};

const PACKET_DOMAIN: &str = "cardinal.judgment-packet.v1";

/// One observation: signed log-ratio toward entity `i`, with the
/// precision (1/variance) it claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PacketObservation {
    pub i: usize,
    pub j: usize,
    pub log_ratio: f64,
    pub precision: f64,
}

/// A sealed bundle of judgments from one party.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentPacket {
    pub version: u32,
    /// The attribute wording (its content IS its identity).
    pub attribute: String,
    /// Prompt template slug the judgments were elicited under.
    pub template: String,
    /// The judging model / party identifier.
    pub judge: String,
    /// (entity id, blake3 hex of the entity text) — sorted by id after
    /// canonicalization. Texts travel as hashes: packets stay light while
    /// pinning exactly what was judged.
    pub entities: Vec<(String, String)>,
    /// Observations, indices into `entities`, canonically sorted.
    pub observations: Vec<PacketObservation>,
    /// Caller-supplied creation stamp (libraries own no clocks).
    pub created: String,
}

/// Errors from packet operations.
#[derive(Debug, thiserror::Error)]
pub enum PacketError {
    #[error("packets disagree on attribute: {0:?} vs {1:?}")]
    AttributeMismatch(String, String),
    #[error("packets disagree on template: {0:?} vs {1:?}")]
    TemplateMismatch(String, String),
    #[error("entity {0} carries different content hashes across packets — tampering or id collision")]
    EntityHashMismatch(String),
    #[error("observation index out of range")]
    BadIndex,
    #[error("no observations to fuse")]
    Empty,
    #[error("solver rejected the fused evidence: {0}")]
    Solver(&'static str),
}

/// Hash an entity's text the way packets pin content.
#[must_use]
pub fn entity_text_hash(text: &str) -> String {
    ContentId::derive("cardinal.entity-text.v1", text.as_bytes()).0
}

impl JudgmentPacket {
    /// Canonicalize in place: entities sorted by id (indices remapped),
    /// observations sorted by full content. Two packets with the same
    /// evidence canonicalize to identical bytes.
    pub fn canonicalize(&mut self) -> Result<(), PacketError> {
        let mut order: Vec<usize> = (0..self.entities.len()).collect();
        order.sort_by(|&a, &b| self.entities[a].0.cmp(&self.entities[b].0));
        let mut remap = vec![0usize; self.entities.len()];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            remap[old_idx] = new_idx;
        }
        self.entities = order.iter().map(|&k| self.entities[k].clone()).collect();
        for ob in &mut self.observations {
            if ob.i >= remap.len() || ob.j >= remap.len() {
                return Err(PacketError::BadIndex);
            }
            ob.i = remap[ob.i];
            ob.j = remap[ob.j];
        }
        self.observations.sort_by(|a, b| {
            (a.i, a.j, a.log_ratio.to_bits(), a.precision.to_bits()).cmp(&(
                b.i,
                b.j,
                b.log_ratio.to_bits(),
                b.precision.to_bits(),
            ))
        });
        Ok(())
    }

    /// Canonical byte encoding: length-prefixed strings, little-endian
    /// f64 bit patterns. No JSON floats anywhere near the identity.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let mut out = Vec::new();
        out.extend_from_slice(&self.version.to_le_bytes());
        put_str(&mut out, &self.attribute);
        put_str(&mut out, &self.template);
        put_str(&mut out, &self.judge);
        out.extend_from_slice(&(self.entities.len() as u64).to_le_bytes());
        for (id, hash) in &self.entities {
            put_str(&mut out, id);
            put_str(&mut out, hash);
        }
        out.extend_from_slice(&(self.observations.len() as u64).to_le_bytes());
        for ob in &self.observations {
            out.extend_from_slice(&(ob.i as u64).to_le_bytes());
            out.extend_from_slice(&(ob.j as u64).to_le_bytes());
            out.extend_from_slice(&ob.log_ratio.to_bits().to_le_bytes());
            out.extend_from_slice(&ob.precision.to_bits().to_le_bytes());
        }
        put_str(&mut out, &self.created);
        out
    }

    /// The packet's content-addressed identity (blake3, domain-tagged).
    /// Any changed byte — one observation, one hash, one character of the
    /// attribute — is a different packet.
    #[must_use]
    pub fn id(&self) -> ContentId {
        ContentId::derive(PACKET_DOMAIN, &self.canonical_bytes())
    }
}

/// The fused posterior: scores over the union entity list, plus the
/// receipts of the fusion itself.
#[derive(Debug, Serialize)]
pub struct FusedPosterior {
    /// Union entities, sorted by id.
    pub entities: Vec<(String, String)>,
    /// Latent scores, aligned to `entities`.
    pub scores: Vec<f64>,
    /// Total cyclic fraction of the fused evidence.
    pub hcr: f64,
    /// The ids of the packets that were fused (sorted).
    pub fused_packet_ids: Vec<String>,
    pub observations: usize,
}

/// Fuse packets into one posterior. Byte-identical for any partition of
/// the same evidence into packets, in any order — the protocol form of
/// the monoid theorem.
pub fn fuse(packets: &[JudgmentPacket]) -> Result<FusedPosterior, PacketError> {
    // Idempotency: dedup by content-addressed packet id, so re-delivery
    // of the same packet is absorbed (set-union merge semantics). Keeps
    // first occurrence; order of survivors is irrelevant because the
    // observation multiset is canonicalized below anyway.
    let mut seen_ids = std::collections::HashSet::new();
    let packets: Vec<&JudgmentPacket> = packets
        .iter()
        .filter(|p| seen_ids.insert(p.id().0))
        .collect();
    let Some(first) = packets.first() else {
        return Err(PacketError::Empty);
    };
    for p in &packets {
        if p.attribute != first.attribute {
            return Err(PacketError::AttributeMismatch(
                first.attribute.clone(),
                p.attribute.clone(),
            ));
        }
        if p.template != first.template {
            return Err(PacketError::TemplateMismatch(
                first.template.clone(),
                p.template.clone(),
            ));
        }
    }

    // Union entities; refuse on content-hash disagreement.
    let mut hash_by_id: HashMap<&str, &str> = HashMap::new();
    for p in &packets {
        for (id, hash) in &p.entities {
            match hash_by_id.get(id.as_str()) {
                Some(existing) if *existing != hash => {
                    return Err(PacketError::EntityHashMismatch(id.clone()));
                }
                _ => {
                    hash_by_id.insert(id, hash);
                }
            }
        }
    }
    let mut entities: Vec<(String, String)> = hash_by_id
        .iter()
        .map(|(id, hash)| ((*id).to_string(), (*hash).to_string()))
        .collect();
    entities.sort();
    let index: HashMap<&str, usize> = entities
        .iter()
        .enumerate()
        .map(|(k, (id, _))| (id.as_str(), k))
        .collect();

    // Collect the observation multiset in union coordinates, tagged by
    // judge (each party is its own rater), then canonicalize.
    let mut all: Vec<(usize, usize, u64, u64, String)> = Vec::new();
    for p in &packets {
        for ob in &p.observations {
            let (Some((id_i, _)), Some((id_j, _))) =
                (p.entities.get(ob.i), p.entities.get(ob.j))
            else {
                return Err(PacketError::BadIndex);
            };
            all.push((
                index[id_i.as_str()],
                index[id_j.as_str()],
                ob.log_ratio.to_bits(),
                ob.precision.to_bits(),
                p.judge.clone(),
            ));
        }
    }
    if all.is_empty() {
        return Err(PacketError::Empty);
    }
    all.sort();

    let mut raters = HashMap::new();
    for (_, _, _, _, judge) in &all {
        raters
            .entry(judge.clone())
            .or_insert_with(RaterParams::default);
    }
    let mut engine = RatingEngine::new(
        entities.len(),
        AttributeParams::default(),
        raters,
        Some(Config::default()),
    )
    .map_err(PacketError::Solver)?;
    let observations: Vec<Observation> = all
        .iter()
        .map(|&(i, j, m_bits, p_bits, ref judge)| {
            let m = f64::from_bits(m_bits);
            let precision = f64::from_bits(p_bits).max(f64::MIN_POSITIVE);
            Observation::from_log_ratio_moments(i, j, m, 1.0 / precision, judge.clone(), 1.0)
        })
        .collect();
    engine.ingest(&observations);
    let summary = engine.solve();

    let mut fused_packet_ids: Vec<String> = packets.iter().map(|p| p.id().0).collect();
    fused_packet_ids.sort();

    Ok(FusedPosterior {
        entities,
        scores: summary.scores,
        hcr: summary.hcr,
        fused_packet_ids,
        observations: all.len(),
    })
}
