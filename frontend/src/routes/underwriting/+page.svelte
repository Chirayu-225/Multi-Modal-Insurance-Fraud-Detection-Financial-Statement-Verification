<script lang="ts">
	import UnderwritingPanel from '$lib/UnderwritingPanel.svelte';
</script>

<main>
	<p><a href="/">&larr; Both sections (claims + underwriting)</a></p>
	<UnderwritingPanel />
</main>

<style>
	main {
		max-width: 680px;
		margin: 2rem auto;
		font-family: system-ui, sans-serif;
		padding: 0 1rem;
	}
</style>
