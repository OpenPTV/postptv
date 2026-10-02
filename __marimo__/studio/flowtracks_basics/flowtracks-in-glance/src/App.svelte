<script lang="ts">
  const notebookLabel = "Flowtracks Basics";
  const viewHeading = "Flowtracks In Glance";
  let resultsVisible = $state(true);
</script>

<div class="page">
  <header class="page-header">
    <div>
      <p>{notebookLabel}</p>
      <h1>{viewHeading}</h1>
    </div>
    <button type="button" onclick={() => (resultsVisible = !resultsVisible)}>
      {resultsVisible ? "Hide results" : "Show results"}
    </button>
  </header>

  {#if resultsVisible}
    <section class="results" aria-label="Notebook results">
      {#each [
        "cell-2",
        "cell-3",
        "cell-4",
        "cell-5",
        "cell-6",
        "cell-7",
        "cell-8",
        "cell-9",
        "cell-10",
        "cell-11",
        "cell-12",
        "cell-13",
        "cell-14",
        "cell-15",
        "cell-16",
      ] as name}
        <marimo-cell {name}></marimo-cell>
      {/each}
    </section>
  {/if}
</div>
