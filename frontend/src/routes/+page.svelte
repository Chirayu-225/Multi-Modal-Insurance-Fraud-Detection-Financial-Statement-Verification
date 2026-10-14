<script lang="ts">
	import { page } from '$app/stores';
	import ClaimsPanel from '$lib/ClaimsPanel.svelte';
	import UnderwritingPanel from '$lib/UnderwritingPanel.svelte';

	type Tab = 'claims' | 'underwriting';

	// Deep-linkable: /?section=underwriting opens straight into that tab,
	// so /underwriting (see its own +page.svelte) can still be bookmarked
	// or shared on its own while this page stays the single hub for both.
	// Read via $page (not window) so this also works during SSR.
	let activeTab = $state<Tab>(
		$page.url.searchParams.get('section') === 'underwriting' ? 'underwriting' : 'claims'
	);
</script>

<main>
	<h1>Multimodal Cross-Source Consistency Verification</h1>
	<p class="subtitle">
		One verification engine, two points in the insurance lifecycle: motor claims
		(<strong>insurtech</strong>) and commercial underwriting (<strong>fintech</strong>). Both run on
		the same cross-source consistency checks -- pick a section below.
	</p>

	<div class="tabs" role="tablist">
		<button
			type="button"
			role="tab"
			aria-selected={activeTab === 'claims'}
			class:active={activeTab === 'claims'}
			onclick={() => (activeTab = 'claims')}
		>
			<span class="kicker">Insurtech</span>
			Motor claims
		</button>
		<button
			type="button"
			role="tab"
			aria-selected={activeTab === 'underwriting'}
			class:active={activeTab === 'underwriting'}
			onclick={() => (activeTab = 'underwriting')}
		>
			<span class="kicker">Fintech</span>
			Commercial underwriting
		</button>
	</div>

	<section class="panel">
		{#if activeTab === 'claims'}
			<ClaimsPanel />
		{:else}
			<UnderwritingPanel />
		{/if}
	</section>
</main>

<style>
	main {
		max-width: 680px;
		margin: 2rem auto;
		font-family: system-ui, sans-serif;
		padding: 0 1rem;
	}
	.subtitle {
		color: #555;
	}
	.tabs {
		display: flex;
		gap: 0.5rem;
		margin-top: 1.5rem;
		border-bottom: 2px solid #eee;
	}
	.tabs button {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.15rem;
		padding: 0.75rem 1rem;
		font-size: 1rem;
		font-weight: 600;
		font-family: inherit;
		background: none;
		border: none;
		border-bottom: 3px solid transparent;
		cursor: pointer;
		color: #777;
	}
	.tabs button:hover {
		color: #111;
	}
	.tabs button.active {
		color: #111;
		border-bottom-color: #2563eb;
	}
	.kicker {
		font-size: 0.7rem;
		font-weight: 700;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: #2563eb;
	}
	.panel {
		margin-top: 1.5rem;
	}
</style>
