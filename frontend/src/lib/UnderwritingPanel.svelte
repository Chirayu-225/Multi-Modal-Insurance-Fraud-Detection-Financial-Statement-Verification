<script lang="ts">
	import {
		submitUnderwritingCase,
		listUnderwritingCases,
		type UnderwritingDetail,
		type UnderwritingSummary
	} from '$lib/api';

	// Phase 1: figures are typed in. Phase 3 replaces this with uploads of
	// the submitted statement (extracted locally) and the annual report
	// (retrieved through the RAG service), feeding the same API shape.
	let companyName = $state('');
	let gstin = $state('');
	let stateCode = $state('');
	let submittedFy = $state('FY2025');
	let submittedRevenue = $state<number | null>(null);
	let submittedProfit = $state<number | null>(null);
	let auditedFy = $state('FY2025');
	let auditedRevenue = $state<number | null>(null);
	let auditedProfit = $state<number | null>(null);
	let opinion = $state<'unqualified' | 'qualified' | 'adverse' | 'disclaimer'>('unqualified');
	let goingConcern = $state(false);

	let submitting = $state(false);
	let error = $state('');
	let result = $state<UnderwritingDetail | null>(null);
	let recent = $state<UnderwritingSummary[]>([]);

	const refresh = () => listUnderwritingCases().then((c) => (recent = c)).catch(() => {});
	refresh();

	function figures(fy: string, revenue: number | null, profit: number | null, source: string) {
		const out = [];
		if (revenue !== null) out.push({ metric: 'revenue' as const, value_inr_crore: revenue, fiscal_year: fy, source });
		if (profit !== null) out.push({ metric: 'net_profit' as const, value_inr_crore: profit, fiscal_year: fy, source });
		return out;
	}

	async function handleSubmit(e: Event) {
		e.preventDefault();
		error = '';
		submitting = true;
		try {
			result = await submitUnderwritingCase({
				company_name: companyName,
				company_gstin: gstin || null,
				registered_state_code: stateCode || null,
				submitted_fiscal_year: submittedFy,
				submitted_figures: figures(submittedFy, submittedRevenue, submittedProfit, 'submitted_statement'),
				audited_fiscal_year: auditedFy || null,
				audited_figures: figures(auditedFy, auditedRevenue, auditedProfit, 'annual_report'),
				auditor_opinion: opinion,
				going_concern_note: goingConcern
			});
			refresh();
		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
		} finally {
			submitting = false;
		}
	}
</script>

<div class="panel-body">
	<h2>Commercial underwriting check</h2>
	<p class="subtitle">
		Enter the applicant's submitted figures and the audited figures from its annual report. The
		backend cross-checks them, along with GSTIN and the auditor's opinion. Amounts in INR crore.
	</p>

	<form onsubmit={handleSubmit}>
		<label>Company name <input type="text" bind:value={companyName} required /></label>
		<div class="row">
			<label>GSTIN <input type="text" bind:value={gstin} maxlength="15" /></label>
			<label>Registered state code <input type="text" bind:value={stateCode} maxlength="2" /></label>
		</div>

		<fieldset>
			<legend>Submitted statement</legend>
			<label>Fiscal year <input type="text" bind:value={submittedFy} required /></label>
			<div class="row">
				<label>Revenue <input type="number" step="0.01" bind:value={submittedRevenue} /></label>
				<label>Net profit <input type="number" step="0.01" bind:value={submittedProfit} /></label>
			</div>
		</fieldset>

		<fieldset>
			<legend>Audited annual report</legend>
			<label>Fiscal year <input type="text" bind:value={auditedFy} /></label>
			<div class="row">
				<label>Revenue <input type="number" step="0.01" bind:value={auditedRevenue} /></label>
				<label>Net profit <input type="number" step="0.01" bind:value={auditedProfit} /></label>
			</div>
			<label>
				Auditor's opinion
				<select bind:value={opinion}>
					<option value="unqualified">Unqualified</option>
					<option value="qualified">Qualified</option>
					<option value="adverse">Adverse</option>
					<option value="disclaimer">Disclaimer</option>
				</select>
			</label>
			<label class="check"><input type="checkbox" bind:checked={goingConcern} /> Going-concern note present</label>
		</fieldset>

		{#if error}<p class="error">{error}</p>{/if}
		<button type="submit" disabled={submitting}>{submitting ? 'Checking…' : 'Run check'}</button>
	</form>

	{#if result?.result}
		<section class="result">
			<h3>
				Risk {result.result.risk_score.toFixed(2)}
				{#if result.result.fast_track}<span class="ok">fast-track</span>{:else}<span class="review">needs review</span>{/if}
			</h3>
			{#if result.result.flags.length === 0}
				<p>No inconsistencies found.</p>
			{:else}
				<ul>
					{#each result.result.flags as f}
						<li><strong>{f.type}</strong> ({f.severity.toFixed(2)}): {f.message}</li>
					{/each}
				</ul>
			{/if}
		</section>
	{/if}

	{#if recent.length}
		<h3>Recent cases</h3>
		<ul class="item-list">
			{#each recent as c}
				<li>
					{c.company_name} — {c.submitted_fiscal_year}
					{#if c.risk_score !== null}
						<span class="score" class:high={c.risk_score > 0.5}>risk {c.risk_score.toFixed(2)}</span>
					{/if}
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.subtitle {
		color: #555;
	}
	form {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		margin: 1.5rem 0;
	}
	fieldset {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		border: 1px solid #ddd;
		padding: 0.75rem;
	}
	.row {
		display: flex;
		gap: 0.75rem;
	}
	.row label {
		flex: 1;
	}
	label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.9rem;
	}
	label.check {
		flex-direction: row;
		align-items: center;
	}
	input,
	select {
		padding: 0.5rem;
		font-size: 1rem;
	}
	button {
		padding: 0.6rem;
		font-size: 1rem;
		cursor: pointer;
	}
	.error {
		color: #b00020;
	}
	.result {
		border-top: 2px solid #eee;
		padding-top: 0.5rem;
	}
	.ok {
		color: #1b7f3b;
		font-size: 0.9rem;
		margin-left: 0.5rem;
	}
	.review {
		color: #b00020;
		font-size: 0.9rem;
		margin-left: 0.5rem;
	}
	.item-list {
		list-style: none;
		padding: 0;
	}
	.item-list li {
		padding: 0.4rem 0;
		border-bottom: 1px solid #eee;
	}
	.score {
		float: right;
		font-weight: 600;
	}
	.score.high {
		color: #b00020;
	}
</style>
