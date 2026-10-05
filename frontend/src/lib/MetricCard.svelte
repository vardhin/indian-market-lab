<script>
  let { label, value = '—', detail = '', tone = 'neutral' } = $props();
</script>

<div class={`metric-card metric-${tone}`}>
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
  {#if detail}<div class="metric-detail">{detail}</div>{/if}
</div>
