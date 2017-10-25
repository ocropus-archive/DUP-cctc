int square(THFloatTensor *input1);
void forward_algorithm(THFloatTensor *lr, THFloatTensor *lmatch, double skip);
void forwardbackward(THFloatTensor *both, THFloatTensor *lmatch);
void ctc_align_targets(THFloatTensor *posteriors, THFloatTensor *outputs, THFloatTensor *targets);
void ctc_align_targets_batch(THFloatTensor *posteriors, THFloatTensor *outputs, THFloatTensor *targets);
