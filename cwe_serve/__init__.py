"""cwe_serve -- service d'inference du classifieur CVE -> CWE (Bloc 5, deploiement / C4).

Charge le DistilBERT fine-tune (71 classes) et expose une prediction avec seuil d'abstention
(le "contrat" : ne repondre que si la confiance depasse le seuil calibre).
"""
