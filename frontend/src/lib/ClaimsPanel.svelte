<script lang="ts">
	import { goto } from '$app/navigation';
	import { submitClaim, listClaims, type ClaimSummary } from '$lib/api';

	let claimantName = $state('');
	let vehicleSegment = $state('hatchback');
	let incidentDate = $state('');
	let incidentType = $state('front_collision');
	let incidentDescription = $state('');
	let photoFile: FileList | undefined = $state();
	let billFile: FileList | undefined = $state();

	let submitting = $state(false);
	let error = $state('');

	let recentClaims = $state<ClaimSummary[]>([]);
	listClaims().then((c) => (recentClaims = c)).catch(() => {});

	async function handleSubmit(e: Event) {
		e.preventDefault();
		error = '';
		if (!photoFile?.[0]) {
			error = 'A damage photo is required.';
			return;
		}
		submitting = true;
		try {
			const form = new FormData();
			form.append('claimant_name', claimantName);
			form.append('vehicle_segment', vehicleSegment);
			form.append('incident_date', incidentDate);
			form.append('described_incident_type', incidentType);
			form.append('incident_description', incidentDescription);
			form.append('photo', photoFile[0]);
			if (billFile?.[0]) form.append('bill', billFile[0]);

			const claim = await submitClaim(form);
			await goto(`/claims/${claim.id}`);
		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
		} finally {
			submitting = false;
		}
	}
</script>

<div class="panel-body">
	<h2>Motor insurance claims</h2>
	<p class="subtitle">
		Submit a claim's photo, description, and repair bill. The backend cross-checks them for
		consistency and returns a risk score with an explanation.
	</p>

	<form onsubmit={handleSubmit}>
		<label>
			Claimant name
			<input type="text" bind:value={claimantName} required />
		</label>

		<label>
			Vehicle segment
			<select bind:value={vehicleSegment}>
				<option value="hatchback">Hatchback</option>
				<option value="sedan">Sedan</option>
				<option value="suv">SUV</option>
			</select>
		</label>

		<label>
			Incident date
			<input type="date" bind:value={incidentDate} required />
		</label>

		<label>
			Described incident type
			<select bind:value={incidentType}>
				<option value="front_collision">Front collision</option>
				<option value="rear_collision">Rear collision</option>
				<option value="left_side_collision">Left side collision</option>
				<option value="right_side_collision">Right side collision</option>
				<option value="parking_scrape">Parking scrape</option>
			</select>
		</label>

		<label>
			Incident description
			<textarea bind:value={incidentDescription} rows="3" required></textarea>
		</label>

		<label>
			Damage photo
			<input type="file" accept="image/*" bind:files={photoFile} required />
		</label>

		<label>
			Repair bill (image, optional)
			<input type="file" accept="image/*" bind:files={billFile} />
		</label>

		{#if error}<p class="error">{error}</p>{/if}

		<button type="submit" disabled={submitting}>
			{submitting ? 'Processing…' : 'Submit claim'}
		</button>
	</form>

	{#if recentClaims.length}
		<h3>Recent claims</h3>
		<ul class="item-list">
			{#each recentClaims as c}
				<li>
					<a href={`/claims/${c.id}`}>
						{c.claimant_name} — {c.described_incident_type} ({c.incident_date})
						{#if c.risk_score !== null}
							<span class="score" class:high={c.risk_score > 0.5}>
								risk {c.risk_score.toFixed(2)}
							</span>
						{/if}
					</a>
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
	label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.9rem;
	}
	input,
	select,
	textarea {
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
