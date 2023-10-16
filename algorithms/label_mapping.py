
def label_mapping_base(logits, mapping_sequence):
    mapping_sequence = mapping_sequence.long()  # Convert to Long type
    modified_logits = logits[:, mapping_sequence]
    return modified_logits
