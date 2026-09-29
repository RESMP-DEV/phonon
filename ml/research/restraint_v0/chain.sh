#!/usr/bin/env bash
# restraint_v0 chain: every remaining variant, train then eval, serial on one GPU.
set -uo pipefail
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
R=research/restraint_v0
L=/data/phonon_restraint_v0/logs
A=/data/phonon_corrector_v0/adapters
GPU=0

hb() { overnight-compute heartbeat --agent opus-restraint --ttl 40m >/dev/null 2>&1; }

# wait for the already-running mid_control training to land
while ! [ -f "$A/restr_mid_control/adapter_config.json" ]; do hb; sleep 60; done
echo "CHAIN: mid_control adapter present $(date +%T)"

ev() { hb; $R/run_eval.sh $GPU 24 16 "$1" > $L/eval_$1.log 2>&1; echo "CHAIN: eval $1 rc=$? $(date +%T)"; }

ev restr_mid_control
echo "STEP e done $(date +%T)"

# d. generic (stage 1 = bigrun_mid_generic, the mid acoustic rows with no user data)
#    then stage 2: real x4 + 25% acoustic replay, 1 epoch, lr 1e-4
hb
$R/run_train.sh $GPU restr_generic_replay /data/phonon_restraint_v0/train_d2.jsonl 1 1e-4 \
  --init-adapter $A/bigrun_mid_generic > $L/train_generic_replay.log 2>&1
echo "CHAIN: train d rc=$? $(date +%T)"
ev restr_generic_replay
echo "STEP d done $(date +%T)"

# a. restraint 50 percent, zero-edit cap 45 percent
hb
$R/run_train.sh $GPU restr_restraint50 /data/phonon_restraint_v0/train_a.jsonl 2 2e-4 \
  > $L/train_restraint50.log 2>&1
echo "CHAIN: train a rc=$? $(date +%T)"
ev restr_restraint50
echo "STEP a done $(date +%T)"

# c. keep-weight 2.0 on copied target tokens (needs --fused-ce off)
hb
$R/run_train.sh $GPU restr_keepweight /data/phonon_bigrun_v0/train_mid.jsonl 2 2e-4 \
  --keep-weight 2.0 --fused-ce off > $L/train_keepweight.log 2>&1
echo "CHAIN: train c rc=$? $(date +%T)"
ev restr_keepweight
echo "STEP c done $(date +%T)"

# b. real rows x12
hb
$R/run_train.sh $GPU restr_real12 /data/phonon_restraint_v0/train_b.jsonl 2 2e-4 \
  > $L/train_real12.log 2>&1
echo "CHAIN: train b rc=$? $(date +%T)"
ev restr_real12
echo "STEP b done $(date +%T)"

echo "CHAIN DONE $(date +%T)"
