// Client API du service CVE -> CWE. Base configurable via VITE_API_URL (defaut : back local).
import type { Prediction } from './types';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8001';

// Extrait le message d'erreur { detail } renvoye par FastAPI, sinon un texte generique.
async function readError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string };
    return body.detail ?? `Erreur ${String(res.status)}`;
  } catch {
    return `Erreur ${String(res.status)}`;
  }
}

// Si l'entree ressemble a du JSON de CVE, on l'envoie tel quel (le backend extrait la description) ;
// sinon on l'envoie comme description brute.
function corps(entree: string): string {
  const t = entree.trim();
  if (t.startsWith('{') || t.startsWith('[')) {
    try {
      return JSON.stringify({ cve: JSON.parse(t) as unknown });
    } catch {
      // pas du JSON valide -> on retombe sur la description brute
    }
  }
  return JSON.stringify({ description: entree });
}

// POST /predict : une description OU un JSON de CVE -> le type de faille (CWE) + contrat d'abstention.
export async function predictCwe(entree: string): Promise<Prediction> {
  const res = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: corps(entree),
  });
  if (!res.ok) {
    throw new Error(await readError(res));
  }
  return (await res.json()) as Prediction;
}
