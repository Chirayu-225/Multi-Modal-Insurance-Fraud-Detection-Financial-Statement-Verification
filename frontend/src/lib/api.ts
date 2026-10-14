// Thin wrapper around the FastAPI backend. Set PUBLIC_API_BASE in
// frontend/.env if the backend isn't at the default (see docker-compose.yml).
const API_BASE = import.meta.env.PUBLIC_API_BASE ?? 'http://localhost:8000';

export interface Flag {
	type: string;
	severity: number;
	message: string;
}

export interface ConsistencyResult {
	risk_score: number;
	model_version: string;
	flags: Flag[];
}

export interface ClaimSummary {
	id: string;
	claimant_name: string;
	incident_date: string;
	described_incident_type: string;
	created_at: string;
	risk_score: number | null;
}

export interface ClaimDetail {
	id: string;
	claimant_name: string;
	vehicle_segment: string;
	incident_date: string;
	described_incident_type: string;
	incident_description: string;
	photo_damage_location: string;
	photo_damage_severity: string;
	photo_capture_date: string | null;
	bill_line_items: { category: string; cost_inr: number }[];
	result: ConsistencyResult | null;
}

export async function submitClaim(formData: FormData): Promise<ClaimDetail> {
	const res = await fetch(`${API_BASE}/claims`, { method: 'POST', body: formData });
	if (!res.ok) {
		const body = await res.text();
		throw new Error(`Submit failed (${res.status}): ${body}`);
	}
	return res.json();
}

export async function listClaims(): Promise<ClaimSummary[]> {
	const res = await fetch(`${API_BASE}/claims`);
	if (!res.ok) throw new Error(`List failed (${res.status})`);
	return res.json();
}

export async function getClaim(id: string): Promise<ClaimDetail> {
	const res = await fetch(`${API_BASE}/claims/${id}`);
	if (!res.ok) throw new Error(`Fetch failed (${res.status})`);
	return res.json();
}

// ---------------------------------------------------------- underwriting

export interface ReportedFigure {
	metric: 'revenue' | 'net_profit' | 'total_assets' | 'total_debt';
	value_inr_crore: number;
	fiscal_year: string;
	source?: string;
}

export interface UnderwritingCaseIn {
	company_name: string;
	company_gstin?: string | null;
	registered_state_code?: string | null;
	submitted_fiscal_year: string;
	submitted_figures: ReportedFigure[];
	audited_fiscal_year?: string | null;
	audited_figures: ReportedFigure[];
	auditor_opinion?: 'unqualified' | 'qualified' | 'adverse' | 'disclaimer' | null;
	going_concern_note: boolean;
}

export interface UnderwritingSummary {
	id: string;
	company_name: string;
	submitted_fiscal_year: string;
	created_at: string;
	risk_score: number | null;
	fast_track: boolean | null;
}

export interface UnderwritingDetail {
	id: string;
	company_name: string;
	submitted_fiscal_year: string;
	audited_fiscal_year: string | null;
	result: (ConsistencyResult & { fast_track: boolean }) | null;
}

export async function submitUnderwritingCase(body: UnderwritingCaseIn): Promise<UnderwritingDetail> {
	const res = await fetch(`${API_BASE}/underwriting/cases`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok) throw new Error(`Submit failed (${res.status}): ${await res.text()}`);
	return res.json();
}

export async function listUnderwritingCases(): Promise<UnderwritingSummary[]> {
	const res = await fetch(`${API_BASE}/underwriting/cases`);
	if (!res.ok) throw new Error(`List failed (${res.status})`);
	return res.json();
}
