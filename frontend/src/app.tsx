import { AlertTriangle, CircleCheck, Loader2, Sparkles } from 'lucide-react';
import { useState } from 'react';

import { predictCwe } from './lib/api';
import type { Prediction, TopHypothese } from './lib/types';

const EXEMPLES: { label: string; texte: string }[] = [
  {
    label: 'XSS',
    texte:
      'Cross-site scripting (XSS) vulnerability in the login page allows remote attackers to inject arbitrary web script or HTML via the username parameter.',
  },
  {
    label: 'Injection SQL',
    texte:
      'SQL injection vulnerability in the search endpoint allows remote attackers to execute arbitrary SQL commands via the id parameter.',
  },
  {
    label: 'Injection commande OS',
    texte:
      'OS command injection vulnerability allows remote attackers to execute arbitrary operating system commands via shell metacharacters in the host parameter.',
  },
  {
    label: 'Traversee de chemin',
    texte:
      'Directory traversal vulnerability allows remote attackers to read arbitrary files via a ../ (dot dot slash) sequence in the file parameter.',
  },
  {
    label: 'Upload de fichier',
    texte:
      'Unrestricted upload of a file with a dangerous type allows remote attackers to execute arbitrary code by uploading a crafted PHP script.',
  },
  {
    label: 'Use-after-free',
    texte:
      'A use-after-free vulnerability allows remote attackers to execute arbitrary code or cause a denial of service via a crafted document that triggers freed memory reuse.',
  },
  {
    label: 'Deserialisation',
    texte:
      'Deserialization of untrusted data allows remote attackers to execute arbitrary code via a crafted serialized object sent to the endpoint.',
  },
  {
    label: 'SSRF',
    texte:
      'Server-side request forgery (SSRF) allows attackers to induce the server to make requests to arbitrary internal hosts via the url parameter.',
  },
  {
    label: 'XXE',
    texte:
      'XML External Entity (XXE) injection allows attackers to read local files or perform SSRF via a crafted XML document with external entity declarations.',
  },
  {
    label: 'Fiche vague',
    texte: 'The product has an issue in some configurations under certain conditions.',
  },
  {
    label: 'JSON de CVE',
    texte:
      '{\n  "id": "CVE-2024-0001",\n  "descriptions": [\n    { "lang": "en", "value": "Use-after-free vulnerability in the rendering engine allows remote attackers to execute arbitrary code via a crafted HTML page." }\n  ]\n}',
  },
];

function Barre({ p, accent }: { p: number; accent: string }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-black/5">
      <div className="h-full rounded-full" style={{ width: `${String(p * 100)}%`, background: accent }} />
    </div>
  );
}

function TopListe({ top, seuil }: { top: TopHypothese[]; seuil: number }) {
  return (
    <ul className="mt-5 flex flex-col gap-3">
      {top.map((h, i) => (
        <li key={h.cwe} className="grid grid-cols-[110px_1fr_56px] items-center gap-3">
          <span className="font-mono text-sm text-muted">{h.cwe}</span>
          <Barre p={h.p} accent={i === 0 ? 'var(--color-pep)' : 'var(--color-faint)'} />
          <span className="text-right font-mono text-sm tabular-nums text-ink">
            {(h.p * 100).toFixed(1)}%
          </span>
        </li>
      ))}
      <li className="mt-1 font-mono text-xs text-faint">
        seuil du contrat : {(seuil * 100).toFixed(1)}%
      </li>
    </ul>
  );
}

function Resultat({ r }: { r: Prediction }) {
  if (r.error !== undefined && r.error !== '') {
    return (
      <div className="mt-6 rounded-xl border border-epss/40 bg-epss-soft px-5 py-4 text-ink">
        {r.error}
      </div>
    );
  }
  const pct = (r.confidence * 100).toFixed(1);
  return (
    <div className="mt-6 rounded-2xl border border-border bg-white p-6 shadow-sm">
      {r.abstain ? (
        <div className="flex items-center gap-3">
          <AlertTriangle className="text-epss" size={26} />
          <div>
            <p className="font-display text-xl font-bold text-ink">Pas assez de confiance pour trancher</p>
            <p className="text-sm text-muted">
              La meilleure hypothese ({r.cwe}, {pct}%) est sous le seuil du contrat. On prefere s'abstenir
              plutot que de router a tort.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3">
          <CircleCheck className="text-good" size={26} />
          <div>
            <p className="font-mono text-xs tracking-widest text-faint uppercase">Type de faille predit</p>
            <p className="font-display text-3xl font-extrabold tracking-tight text-pep">{r.cwe}</p>
            <p className="text-sm text-muted">
              Confiance {pct}% -- au-dessus du seuil du contrat, on tranche.
            </p>
          </div>
        </div>
      )}
      <TopListe top={r.top} seuil={r.threshold} />
    </div>
  );
}

export function App() {
  const [texte, setTexte] = useState('');
  const [chargement, setChargement] = useState(false);
  const [erreur, setErreur] = useState('');
  const [resultat, setResultat] = useState<Prediction | undefined>(undefined);

  async function classer() {
    setChargement(true);
    setErreur('');
    setResultat(undefined);
    try {
      setResultat(await predictCwe(texte));
    } catch (e) {
      setErreur(e instanceof Error ? e.message : 'Erreur inconnue');
    } finally {
      setChargement(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <header className="mb-8">
        <p className="mb-3 flex items-center gap-2 font-mono text-xs tracking-widest text-pep uppercase">
          <Sparkles size={14} /> Deep learning -- DistilBERT fine-tune
        </p>
        <h1 className="font-display text-4xl font-extrabold tracking-tight text-ink">CVE &rarr; CWE</h1>
        <p className="mt-2 max-w-xl text-muted">
          Collez la <strong>description</strong> d'une vulnerabilite (ou le <strong>JSON</strong> complet
          d'une CVE) : le modele predit son <strong>type de faille</strong> (CWE), avec sa confiance -- et{' '}
          <strong>s'abstient</strong> quand il n'est pas assez sur.
        </p>
      </header>

      <div className="flex flex-wrap gap-2">
        {EXEMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            onClick={() => {
              setTexte(ex.texte);
            }}
            className="rounded-lg border border-border bg-white px-3 py-1.5 font-mono text-xs text-muted transition hover:border-pep hover:text-pep"
          >
            {ex.label}
          </button>
        ))}
      </div>

      <textarea
        value={texte}
        onChange={(e) => {
          setTexte(e.target.value);
        }}
        rows={5}
        placeholder="Ex : Cross-site scripting (XSS) vulnerability in...  (ou collez un JSON de CVE)"
        className="mt-3 w-full resize-y rounded-xl border border-border bg-white p-4 text-ink outline-none focus:border-pep"
      />

      <button
        type="button"
        onClick={() => void classer()}
        disabled={chargement || texte.trim().length < 10}
        className="mt-3 inline-flex items-center gap-2 rounded-xl bg-pep px-5 py-2.5 font-medium text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {chargement ? <Loader2 className="animate-spin" size={18} /> : undefined}
        {chargement ? 'Classement...' : 'Classer'}
      </button>

      {erreur !== '' && (
        <div className="mt-6 rounded-xl border border-epss/40 bg-epss-soft px-5 py-4 text-ink">
          {erreur}{' '}
          <span className="text-muted">(l'API est-elle lancee sur le port 8001 ?)</span>
        </div>
      )}

      {resultat !== undefined && <Resultat r={resultat} />}

      <footer className="mt-10 border-t border-border pt-5 font-mono text-xs text-faint">
        Contrat (test 2025) : ~81,5 % de precision sur 57,9 % du flux -- le seuil arbitre precision vs
        couverture. 71 classes (70 CWE frequents + CWE-OTHER).
      </footer>
    </main>
  );
}
