use cardinal_harness::cache::PairwiseCacheKey;

#[test]
fn pairwise_cache_key_is_stable_and_sensitive_to_inputs() {
    let key1 = PairwiseCacheKey::new(
        "openai/gpt-5-mini",
        "canonical_v2",
        "template_hash",
        "clarity",
        "clarity of explanation",
        "a",
        "Entity A text",
        "b",
        "Entity B text",
    );
    let key2 = PairwiseCacheKey::new(
        "openai/gpt-5-mini",
        "canonical_v2",
        "template_hash",
        "clarity",
        "clarity of explanation",
        "a",
        "Entity A text",
        "b",
        "Entity B text",
    );

    assert_eq!(key1.key_hash, key2.key_hash);
    assert_eq!(key1.attribute_prompt_hash, key2.attribute_prompt_hash);
    assert_eq!(key1.entity_a_hash, key2.entity_a_hash);
    assert_eq!(key1.entity_b_hash, key2.entity_b_hash);

    let key3 = PairwiseCacheKey::new(
        "openai/gpt-5-mini",
        "canonical_v2",
        "template_hash",
        "clarity",
        "clarity of explanation",
        "a",
        "Entity A text (changed)",
        "b",
        "Entity B text",
    );
    assert_ne!(key1.key_hash, key3.key_hash);
    assert_ne!(key1.entity_a_hash, key3.entity_a_hash);
    assert_eq!(key1.entity_b_hash, key3.entity_b_hash);
}

#[test]
fn pairwise_cache_key_depends_on_entity_order() {
    let ab = PairwiseCacheKey::new(
        "openai/gpt-5-mini",
        "canonical_v2",
        "template_hash",
        "clarity",
        "clarity of explanation",
        "a",
        "Entity A text",
        "b",
        "Entity B text",
    );
    let ba = PairwiseCacheKey::new(
        "openai/gpt-5-mini",
        "canonical_v2",
        "template_hash",
        "clarity",
        "clarity of explanation",
        "b",
        "Entity B text",
        "a",
        "Entity A text",
    );
    assert_ne!(ab.key_hash, ba.key_hash);
}
