<script lang="ts">
	import { getClaim, type ClaimDetail } from '$lib/api';

	let { data }: { data: { id: string } } = $props();

	let claim = $state<ClaimDetail | null>(null);
	let error = $state('');

	getClaim(data.id)
		.then((c) => (claim = c))
		.catch((e) => (error = e instanceof Error ? e.message : String(e)));
</script>

<main>
	<p><a href="/">&larr; back</a></p>

	{#if error}
		<p class="error">{error}</p>
	{:else if !claim}
		<p>Loading…</p>
	{:else}
		<h1>Claim: {claim.claimant_name}</h1>
		<p>
			{claim.described_incident_type} on {claim.incident_date} ·
			{claim.vehicle_segment}
		</p>
		<p class="desc">{claim.incident_description}</p>

		<section class="evidence">
			<div><strong>Photo shows:</strong> {claim.photo_damage_location} ({claim.photo_damage_severity})</div>
			<div><strong>Photo capture date:</strong> {claim.photo_capture_date ?? 'unavailable'}</div>
			<div>
				<strong>Bill items:</strong>
				<ul>
					{#each claim.bill_line_items as item}
						<li>{item.category}: &#8377;{item.cost_inr.toLocaleString()}</li>
					{/each}
				</ul>
			</div>
		</section>

		{#if claim.result}
			<section class="result" class:high={claim.result.risk_score > 0.5}>
				<h2>Risk score: {claim.result.risk_score.toFixed(2)}</h2>
				<p class="model-version">model: {claim.result.model_version}</p>

				{#if claim.result.flags.length === 0}
					<p>No inconsistencies detected.</p>
				{:else}
					<ul class="flags">
						{#each claim.result.flags as flag}
							<li>
								<span class="flag-type">{flag.type}</span>
								<span class="flag-severity">severity {flag.severity.toFixed(2)}</span>
								<p>{flag.message}</p>
							</li>
						{/each}
					</ul>
				{/if}
			</section>
		{/if}
	{/if}
</main>

<style>
	main {
		max-width: 640px;
		margin: 2rem auto;
		font-family: system-ui, sans-serif;
		padding: 0 1rem;
	}
	.desc {
		color: #444;
		font-style: italic;
	}
	.evidence {
		margin: 1rem 0;
		padding: 1rem;
		background: #f7f7f7;
		border-radius: 8px;
	}
	.result {
		margin-top: 1.5rem;
		padding: 1rem;
		border-radius: 8px;
		border: 2px solid #2e7d32;
	}
	.result.high {
		border-color: #b00020;
	}
	.model-version {
		color: #888;
		font-size: 0.8rem;
	}
	.flags {
		list-style: none;
		padding: 0;
	}
	.flags li {
		padding: 0.5rem 0;
		border-bottom: 1px solid #eee;
	}
	.flag-type {
		font-weight: 600;
		text-transform: uppercase;
		font-size: 0.75rem;
		margin-right: 0.5rem;
	}
	.flag-severity {
		color: #b00020;
		font-size: 0.8rem;
	}
	.error {
		color: #b00020;
	}
</style>
