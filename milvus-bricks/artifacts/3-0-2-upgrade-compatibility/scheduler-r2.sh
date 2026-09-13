#!/usr/bin/env bash
set -euo pipefail

export ARGO_SERVER='argo-workflows.zilliz.cc:443'
export ARGO_HTTP1=true
export ARGO_SECURE=true
export ARGO_BASE_HREF=
export ARGO_TOKEN=''
export ARGO_NAMESPACE=qa

out_dir='artifacts/3-0-2-upgrade-compatibility/rendered'
label='release-validation=milvus-3-0-2-20260913-r2'
standalone_queue=(
  standalone-3-0-1-vortex-self-compat-upgrade-rollback
  standalone-3-0-0-to-3-0-1-vortex-enable-rollback
  standalone-3-0-1-json-shredding-vortex-rollback
  standalone-3-0-1-loon-ffi-rollback
  standalone-3-0-1-vortex-disable-rollback
  standalone-3-0-1-vortex-disable-keep-loon-rollback
)
cluster_queue=(
  cluster-3-0-index-v11-v4-upgrade-rollback
  cluster-3-0-1-vortex-self-compat-upgrade-rollback
  cluster-3-0-0-to-3-0-1-vortex-enable-rollback
  cluster-3-0-1-json-shredding-vortex-rollback
  cluster-3-0-1-loon-ffi-rollback
  cluster-3-0-baseline-to-3-0-latest-json-shredding-rollback-3-0-baseline
  cluster-3-0-baseline-to-3-0-latest-woodpecker-2cu-ha-rollback-3-0-baseline
)

submit_scenario() {
  local scenario="$1" prefix="$2" json="$out_dir/$scenario.json"
  local template key value workflow
  local -a params
  template="$(jq -r '.workflow_template' "$json")"
  params=()
  while IFS=$'\t' read -r key value; do
    params+=('-p' "$key=$value")
  done < <(jq -r '.parameters | to_entries[] | [.key, .value] | @tsv' "$json")
  workflow="$(argo submit -n qa \
    --from "workflowtemplate/$template" \
    --generate-name "$prefix" \
    --labels "$label" \
    "${params[@]}" \
    -p keep-milvus=false \
    -o name \
    --request-timeout 30s)"
  printf '%s SUBMITTED %s scenario=%s\n' "$(date '+%F %T')" "$workflow" "$scenario"
}

while true; do
  listing="$(argo list -n qa -l "$label" -o json --request-timeout 30s)"
  active_st="$(jq '[.[] | select((.status.phase=="Running" or .status.phase=="Pending") and ([.spec.arguments.parameters[]? | select(.name=="scenario-id") | .value][0] | startswith("standalone-")))] | length' <<<"$listing")"
  active_cl="$(jq '[.[] | select((.status.phase=="Running" or .status.phase=="Pending") and ([.spec.arguments.parameters[]? | select(.name=="scenario-id") | .value][0] | startswith("cluster-")))] | length' <<<"$listing")"
  for scenario in "${standalone_queue[@]}"; do
    ((active_st >= 4)) && break
    exists="$(jq --arg s "$scenario" '[.[] | select([.spec.arguments.parameters[]? | select(.name=="scenario-id") | .value][0] == $s)] | length' <<<"$listing")"
    if ((exists == 0)); then
      submit_scenario "$scenario" 'r302-snext-'
      active_st=$((active_st + 1))
    fi
  done
  for scenario in "${cluster_queue[@]}"; do
    ((active_cl >= 3)) && break
    exists="$(jq --arg s "$scenario" '[.[] | select([.spec.arguments.parameters[]? | select(.name=="scenario-id") | .value][0] == $s)] | length' <<<"$listing")"
    if ((exists == 0)); then
      submit_scenario "$scenario" 'r302-cnext-'
      active_cl=$((active_cl + 1))
    fi
  done
  listing="$(argo list -n qa -l "$label" -o json --request-timeout 30s)"
  total="$(jq 'length' <<<"$listing")"
  running="$(jq '[.[] | select(.status.phase=="Running" or .status.phase=="Pending")] | length' <<<"$listing")"
  succeeded="$(jq '[.[] | select(.status.phase=="Succeeded")] | length' <<<"$listing")"
  failed="$(jq '[.[] | select(.status.phase=="Failed" or .status.phase=="Error")] | length' <<<"$listing")"
  printf '%s SUMMARY total=%s active=%s standalone=%s cluster=%s succeeded=%s failed=%s\n' \
    "$(date '+%F %T')" "$total" "$running" "$active_st" "$active_cl" "$succeeded" "$failed"
  if ((total >= 20 && running == 0)); then
    jq -r '.[] | [[.spec.arguments.parameters[]? | select(.name=="scenario-id") | .value][0], .metadata.name, .status.phase, .status.startedAt, .status.finishedAt] | @tsv' <<<"$listing" | sort
    exit 0
  fi
  sleep 30
done
