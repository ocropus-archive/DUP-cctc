// #define SAFE
// #undef NDEBUG
#include <iostream>
#include <future>
#include <vector>
#include <thread>
#include "minitensor.h"
extern "C" {
#include <omp.h>
#include <math.h>
#include <assert.h>
#include "cctc.h"
}
#include "helper.h"

#ifndef MAXEXP
#define MAXEXP 30
#endif

int square(THFloatTensor *input1) {
    TFloat a(input1, "square");
    if(a.dim()!=2) return 0;
    for(int i=0; i<a.size(0); i++)
        for(int j=0; j<a.size(1); j++)
            a(i,j) = a(i,j) * a(i,j);
    return 1;
}

inline int rows(TFloat &m) {
    if (m.dim()!=2) abort();
    return m.size(0);
}

inline int cols(TFloat &m) {
    if (m.dim()!=2) abort();
    return m.size(1);
}

inline float limexp(float x) {
  if (x < -MAXEXP) return exp(-MAXEXP);
  if (x > MAXEXP) return exp(MAXEXP);
  return exp(x);
}

inline float log_add(float x, float y) {
  if (fabs(x - y) > 10) return fmax(x, y);
  return log(exp(x - y) + 1) + y;
}

inline float log_mul(float x, float y) { return x + y; }

bool check_rownorm(TFloat &a) {
  for(int i=0; i<a.size(0); i++) {
    double total = 0.0;
    for(int j=0; j<a.size(1); j++) {
      double value = a(i,j);
      if (value<0) return false;
      if (value>1) return false;
      total += value;
    }
    if (abs(total-1.0) > 1e-4) return false;
  }
  return true;
}

static void forward_algorithm(TFloat &lr, TFloat &lmatch, double skip = -5) {
  int n = rows(lmatch), m = cols(lmatch);
  lr.resize(n, m).fill(0);
  TFloat v(m), w(m);
  for (int j = 0; j < m; j++) v(j) = skip * j;
  for (int i = 0; i < n; i++) {
    w(0) = skip * i;
    for (int j = 1; j < m; j++) w(j) = v(j - 1);
    for (int j = 0; j < m; j++) {
      float same = log_mul(v(j), lmatch(i, j));
      float next = log_mul(w(j), lmatch(i, j));
      v(j) = log_add(same, next);
    }
    for (int j = 0; j < m; j++) lr(i, j) = v(j);
  }
}

static void forwardbackward(TFloat &both, TFloat &lmatch) {
  int n = rows(lmatch), m = cols(lmatch);
  TFloat lr;
  forward_algorithm(lr, lmatch);
  TFloat rlmatch(n, m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) rlmatch(i, j) = lmatch(n - i - 1, m - j - 1);
  TFloat rrl;
  forward_algorithm(rrl, rlmatch);
  TFloat rl(n, m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) rl(i, j) = rrl(n - i - 1, m - j - 1);
  both = lr + rl;
}

void ctc_align_targets(TFloat &posteriors, TFloat &outputs,
                       TFloat &targets) {
  assert(cols(targets) == cols(outputs));
  assert(rows(targets) <= rows(outputs));
  assert(check_rownorm(outputs));
  assert(check_rownorm(targets));
  double lo = 1e-6;
  int n1 = rows(outputs);
  int n2 = rows(targets);
  int nc = cols(targets);

  // compute log probability of state matches
  TFloat lmatch;
  lmatch.resize(n1, n2).fill(0);
  for (int t1 = 0; t1 < n1; t1++) {
    TFloat out(nc);
    for (int i = 0; i < nc; i++) out(i) = fmax(lo, outputs(t1, i));
    out = out / out.sum();
    for (int t2 = 0; t2 < n2; t2++) {
      double total = 0.0;
      for (int k = 0; k < nc; k++) total += out(k) * targets(t2, k);
      lmatch(t1, t2) = log(total);
    }
  }
  // compute unnormalized forward backward algorithm
  TFloat both;
  forwardbackward(both, lmatch);

  // compute normalized state probabilities
  TFloat epath = both - both.max();
  for(int i=0; i<epath.size(0); i++) 
      for(int j=0; j<epath.size(1); j++) 
          epath(i, j) = limexp(epath(i, j));
  for (int j = 0; j < n2; j++) {
    double total = 0.0;
    for (int i = 0; i < rows(epath); i++) total += epath(i, j);
    total = fmax(1e-9, total);
    for (int i = 0; i < rows(epath); i++) epath(i, j) /= total;
  }

  // compute posterior probabilities for each class and normalize
  TFloat aligned;
  aligned.resize(n1, nc).fill(0);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < nc; j++) {
      double total = 0.0;
      for (int k = 0; k < n2; k++) {
        double value = epath(i, k) * targets(k, j);
        total += value;
      }
      aligned(i, j) = total;
    }
  }
  for (int i = 0; i < n1; i++) {
    double total = 0.0;
    for (int j = 0; j < nc; j++) total += aligned(i, j);
    total = fmax(total, 1e-9);
    for (int j = 0; j < nc; j++) aligned(i, j) /= total;
  }
  assert(check_rownorm(aligned));
  posteriors.copy(aligned);
}

void forward_algorithm(THFloatTensor *lr, THFloatTensor *lmatch, double skip = -5) {
    TFloat lr_(lr, "lr1");
    TFloat lmatch_(lmatch, "lmatch1");
    forward_algorithm(lr_, lmatch_, skip);
}

void forwardbackward(THFloatTensor *both, THFloatTensor *lmatch) {
    TFloat both_(both, "both1");
    TFloat lmatch_(lmatch, "lmatch1");
    forwardbackward(both_, lmatch_);
}

void ctc_align_targets(THFloatTensor *posteriors, THFloatTensor *outputs, THFloatTensor *targets) {
    TFloat outputs_(outputs, "outputs1");
    TFloat targets_(targets, "targets1");
    TFloat posteriors_(posteriors, "posteriors1");
    posteriors_.resizeAs(outputs_);
    ctc_align_targets(posteriors_, outputs_, targets_);
}

void ctc_align_targets_batch(THFloatTensor *posteriors, THFloatTensor *outputs, THFloatTensor *targets) {
    TFloat outputs_(outputs, "outputs");
    TFloat targets_(targets, "targets");
    assert(outputs_.dim()==3);
    assert(targets_.dim()==3);
    TFloat posteriors_(posteriors, "posteriors");
    posteriors_.resizeAs(outputs_);
    if(getenv("CTC_NOTHREAD") && atoi(getenv("CTC_NOTHREAD"))) {
        for(int i=0; i<outputs_.size(0); i++) {
            TFloat p = posteriors_.select(0, i);
            TFloat o = outputs_.select(0, i);
            TFloat t = targets_.select(0, i);
            ctc_align_targets(p, o, t);
        }
    } else {
        int bs = posteriors_.size(0);
        std::vector<std::future<int> > results(bs);
        for(int i=0; i<outputs_.size(0); i++) {
            results[i] = std::async(
                std::launch::async,
                [i, &posteriors_, &outputs_, &targets_]() {
                    TFloat p = posteriors_.select(0, i); p.note = "p";
                    TFloat o = outputs_.select(0, i); p.note = "o";
                    TFloat t = targets_.select(0, i); p.note = "t";
                    ctc_align_targets(p, o, t);
                    return 1;
                });
        }
        for(int i=0; i<outputs_.size(0); i++) {
            results[i].wait();
        }
    }
}
