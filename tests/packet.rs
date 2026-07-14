//! The judgment packet protocol, pinned at the only tolerance worthy of
//! it: byte identity. Any partition of the same evidence into packets,
//! fused in any order, must produce bit-for-bit the same posterior as a
//! single party holding everything — the monoid theorem in protocol
//! form. Plus: tamper-evidence (one flipped byte = a different packet;
//! conflicting entity content = refusal to fuse) and serde round-trip.

use cardinal_harness::packet::{
    entity_text_hash, fuse, JudgmentPacket, PacketError, PacketObservation,
};

fn entities(n: usize) -> Vec<(String, String)> {
    (0..n)
        .map(|k| {
            let text = format!("entity text number {k}");
            (format!("e{k:02}"), entity_text_hash(&text))
        })
        .collect()
}

/// Deterministic evidence over n entities: every pair, slightly noisy
/// log-ratios from a planted latent, mixed precisions.
fn evidence(n: usize, seed: u64) -> Vec<PacketObservation> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut obs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            obs.push(PacketObservation {
                i,
                j,
                log_ratio: (i as f64 - j as f64) * 0.4 + 0.05 * next(),
                precision: 1.0 + 0.5 * next().abs(),
            });
        }
    }
    obs
}

fn packet(judge: &str, ents: Vec<(String, String)>, obs: Vec<PacketObservation>) -> JudgmentPacket {
    let mut p = JudgmentPacket {
        version: 1,
        attribute: "depth of insight about living well".into(),
        template: "canonical_v2".into(),
        judge: judge.into(),
        entities: ents,
        observations: obs,
        created: "2026-07-07".into(),
    };
    p.canonicalize().unwrap();
    p
}

#[test]
fn any_partition_fuses_byte_identical_to_the_whole() {
    let ents = entities(7);
    let all = evidence(7, 42);

    // Single party holding everything.
    let whole = fuse(&[packet("party", ents.clone(), all.clone())]).unwrap();

    // Two parties, arbitrary split — same judge tag so the evidence
    // multiset is literally identical to the whole.
    let (a_obs, b_obs): (Vec<_>, Vec<_>) =
        all.iter()
            .cloned()
            .enumerate()
            .fold((vec![], vec![]), |(mut a, mut b), (k, ob)| {
                if k % 3 == 0 {
                    a.push(ob);
                } else {
                    b.push(ob);
                }
                (a, b)
            });
    let pa = packet("party", ents.clone(), a_obs);
    let pb = packet("party", ents.clone(), b_obs);

    let ab = fuse(&[pa.clone(), pb.clone()]).unwrap();
    let ba = fuse(&[pb, pa]).unwrap();

    for ((w, x), y) in whole
        .scores
        .iter()
        .zip(ab.scores.iter())
        .zip(ba.scores.iter())
    {
        assert_eq!(
            w.to_bits(),
            x.to_bits(),
            "two-party fusion must be BYTE-identical to single-party"
        );
        assert_eq!(x.to_bits(), y.to_bits(), "and independent of packet order");
    }
    assert_eq!(whole.hcr.to_bits(), ab.hcr.to_bits());
    assert_eq!(ab.observations, whole.observations);
    assert_eq!(whole.diag_cov.len(), whole.entities.len());
    for ((w, x), y) in whole
        .diag_cov
        .iter()
        .zip(ab.diag_cov.iter())
        .zip(ba.diag_cov.iter())
    {
        assert_eq!(
            w.to_bits(),
            x.to_bits(),
            "uncertainty must be byte-identical across evidence partitions"
        );
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "uncertainty must be independent of packet order"
        );
    }
}

#[test]
fn partial_overlap_of_entity_sets_fuses_on_the_union() {
    // Party A judged e00..e04, party B judged e02..e06: fusion lands on
    // the 7-entity union with both parties' evidence in one graph.
    let all_ents = entities(7);
    let a_ents: Vec<_> = all_ents[..5].to_vec();
    let b_ents: Vec<_> = all_ents[2..].to_vec();
    let a_obs = vec![
        PacketObservation {
            i: 0,
            j: 1,
            log_ratio: 0.4,
            precision: 1.0,
        },
        PacketObservation {
            i: 1,
            j: 2,
            log_ratio: 0.4,
            precision: 1.0,
        },
        PacketObservation {
            i: 2,
            j: 3,
            log_ratio: 0.4,
            precision: 1.0,
        },
        PacketObservation {
            i: 3,
            j: 4,
            log_ratio: 0.4,
            precision: 1.0,
        },
    ];
    let b_obs = vec![
        PacketObservation {
            i: 0,
            j: 1,
            log_ratio: 0.4,
            precision: 1.0,
        },
        PacketObservation {
            i: 1,
            j: 2,
            log_ratio: 0.4,
            precision: 1.0,
        },
        PacketObservation {
            i: 2,
            j: 3,
            log_ratio: 0.4,
            precision: 1.0,
        },
        PacketObservation {
            i: 3,
            j: 4,
            log_ratio: 0.4,
            precision: 1.0,
        },
    ];
    let fused = fuse(&[packet("alice", a_ents, a_obs), packet("bob", b_ents, b_obs)]).unwrap();
    assert_eq!(fused.entities.len(), 7, "union of both parties' worlds");
    // The chain is consistent: scores strictly decreasing along e00..e06.
    for w in fused.scores.windows(2) {
        assert!(
            w[0] > w[1],
            "chained evidence must order the union: {:?}",
            fused.scores
        );
    }
}

#[test]
fn one_flipped_byte_is_a_different_packet() {
    let p1 = packet("party", entities(4), evidence(4, 7));
    let mut p2 = p1.clone();
    let id1 = p1.id();
    // Tamper: nudge one observation by one ulp.
    p2.observations[0].log_ratio = f64::from_bits(p2.observations[0].log_ratio.to_bits() ^ 1);
    assert_ne!(id1, p2.id(), "identity must see a single flipped bit");
    // And the attribute wording is identity-bearing too.
    let mut p3 = p1.clone();
    p3.attribute.push(' ');
    assert_ne!(id1, p3.id());
}

#[test]
fn conflicting_entity_content_refuses_to_fuse() {
    let ents_a = entities(3);
    let mut ents_b = entities(3);
    // Same id, different underlying text: an impersonated entity.
    ents_b[1].1 = entity_text_hash("something else entirely");
    let pa = packet("alice", ents_a, evidence(3, 1));
    let pb = packet("bob", ents_b, evidence(3, 2));
    match fuse(&[pa, pb]) {
        Err(PacketError::EntityHashMismatch(id)) => assert_eq!(id, "e01"),
        other => panic!("must refuse impersonated entities: {other:?}"),
    }
}

#[test]
fn packets_round_trip_through_json_with_identity_intact() {
    let p = packet("party", entities(5), evidence(5, 99));
    let id = p.id();
    let json = serde_json::to_string(&p).unwrap();
    let back: JudgmentPacket = serde_json::from_str(&json).unwrap();
    assert_eq!(
        id,
        back.id(),
        "serialization must not perturb identity (canonical bytes are \
         float-bit-exact; JSON is transport only)"
    );
}

#[test]
fn redelivered_packet_is_absorbed_fusion_is_idempotent() {
    // The CRDT claim, made testable (red team 2026-07-09: pre-fix, a
    // duplicate packet double-counted its observations — commutative,
    // associative, NOT idempotent). Merge must be set union over
    // content-addressed packets: fuse([a, b, a]) == fuse([a, b]),
    // byte-identical, and the duplicate id appears once in provenance.
    let ents = entities(6);
    let all = evidence(6, 7);
    let (a_obs, b_obs): (Vec<_>, Vec<_>) =
        all.iter()
            .cloned()
            .enumerate()
            .fold((vec![], vec![]), |(mut a, mut b), (k, ob)| {
                if k % 2 == 0 {
                    a.push(ob);
                } else {
                    b.push(ob);
                }
                (a, b)
            });
    let pa = packet("alice", ents.clone(), a_obs);
    let pb = packet("bob", ents.clone(), b_obs);

    let clean = fuse(&[pa.clone(), pb.clone()]).unwrap();
    let redelivered = fuse(&[pa.clone(), pb.clone(), pa.clone()]).unwrap();

    for (x, y) in clean.scores.iter().zip(redelivered.scores.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "re-delivered packet must be absorbed byte-identically"
        );
    }
    assert_eq!(clean.observations, redelivered.observations);
    assert_eq!(clean.fused_packet_ids, redelivered.fused_packet_ids);
    assert_eq!(
        redelivered
            .fused_packet_ids
            .iter()
            .filter(|id| **id == pa.id().0)
            .count(),
        1,
        "duplicate id must appear once in provenance"
    );
}
