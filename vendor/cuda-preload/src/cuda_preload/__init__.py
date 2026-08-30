"""Précharge les libs CUDA fournies par les wheels ``nvidia-*``.

Contourne un bug d'empaquetage de TensorFlow 2.21 : le RUNPATH de
``libtensorflow_cc.so.2`` ne référence que ``nvidia/cusolver/lib`` (répété 36
fois) au lieu des 11 répertoires CUDA. TF n'arrive donc pas à ``dlopen``
libcudart / libcudnn / libcublas, log ``Cannot dlopen some GPU libraries``,
puis ``Skipping registering GPU devices`` — et bascule silencieusement en CPU.

En chargeant ces ``.so`` avec ``RTLD_GLOBAL`` au démarrage de l'interpréteur,
les ``dlopen`` ultérieurs de TF retrouvent les objets déjà chargés via leur
soname. On ne dépend pas de ``LD_LIBRARY_PATH``, qui est lu par ``ld.so`` au
lancement du process et ne peut donc pas être corrigé depuis un notebook déjà
démarré — ni depuis l'``env`` d'un kernelspec que VS Code contourne en lançant
l'interpréteur du venv directement.

Déclenché automatiquement par ``zzz_cuda_preload.pth``, installé à la racine
du site-packages par ce même paquet.
"""

import ctypes
import glob
import os
import sys

__all__ = ["preload"]

# L'ordre reflète les dépendances : une lib doit être chargée après celles
# dont elle dépend (cudnn -> cublas, cusolver -> cusparse/nvJitLink).
_LIBS = (
    "libcudart.so.*",
    "libnvJitLink.so.*",
    "libcublasLt.so.*",
    "libcublas.so.*",
    "libcufft.so.*",
    "libcurand.so.*",
    "libcusparse.so.*",
    "libcusolver.so.*",
    "libcudnn_graph.so.*",
    "libcudnn_engines_precompiled.so.*",
    "libcudnn_engines_runtime_compiled.so.*",
    "libcudnn_heuristic.so.*",
    "libcudnn_ops.so.*",
    "libcudnn_adv.so.*",
    "libcudnn_cnn.so.*",
    "libcudnn.so.*",
    "libnccl.so.*",
    "libcupti.so.*",
)


def _nvidia_root():
    """Répertoire des wheels nvidia-*, tel qu'installé dans ce venv."""
    return os.path.join(
        sys.prefix,
        "lib",
        "python%d.%d" % sys.version_info[:2],
        "site-packages",
        "nvidia",
    )


def preload():
    """Charge les libs CUDA trouvées. Renvoie la liste des chemins chargés."""
    root = _nvidia_root()
    loaded = []
    if not os.path.isdir(root):
        return loaded
    for pattern in _LIBS:
        for path in sorted(glob.glob(os.path.join(root, "*", "lib", pattern))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Lib absente ou incompatible : on dégrade vers le CPU.
                continue
            loaded.append(path)
    return loaded
