// Types de la reponse du service CVE -> CWE.

export type TopHypothese = {
  cwe: string;
  p: number;
};

export type Prediction = {
  cwe: string;
  confidence: number;
  abstain: boolean;
  threshold: number;
  top: TopHypothese[];
  error?: string;
};
