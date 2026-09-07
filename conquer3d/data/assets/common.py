"""Standard 3D benchmark model assets loader.

This module provides automatic download and loading interfaces for standard
geometry processing benchmark models (Stanford Bunny, Dragon, Armadillo, Cow, Spot, etc.).
"""

from typing import Tuple, Optional
import os
import urllib.request
import torch
from conquer3d.io.obj import read_obj


class Common3D:
    """Base class for benchmark test models from the common-3d-test-models repository.

    Downloads the specific `.obj` model directly via the raw GitHub URL
    and loads its vertices, faces, and optional vertex colors.

    Attributes:
        filename (str): Name of the remote `.obj` file.
        download_dir (str): Local cache directory.
        vertices (torch.Tensor | None): Float32 tensor of shape `(V, 3)`.
        faces (torch.Tensor | None): Int64 tensor of shape `(F, 3)`.
        colors (torch.Tensor | None): Optional float32 tensor of shape `(V, 3)`.
    """

    def __init__(self, filename: str, download_dir: str = "~/.conquer3d", verbose: bool = False) -> None:
        """Initializes and automatically downloads/caches the model.

        Args:
            filename (str): Name of the `.obj` asset.
            download_dir (str, optional): Target local directory. Defaults to `"~/.conquer3d"`.
            verbose (bool, optional): Whether to print download progress. Defaults to False.
        """
        self.filename = filename
        self.url = f"https://raw.githubusercontent.com/KhoiDOO/common-3d-test-models/master/data/{self.filename}"
        self.download_dir = os.path.expanduser(download_dir)
        self.vertices = None
        self.faces = None
        self.colors = None
        self.verbose = verbose

        self._load()

    def _load(self) -> None:
        os.makedirs(self.download_dir, exist_ok=True)
        obj_path = os.path.join(self.download_dir, self.filename)
        
        if not os.path.exists(obj_path):
            if self.verbose:
                print(f"Downloading {self.filename} from {self.url}...")
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(obj_path, 'wb') as out_file:
                out_file.write(response.read())
            if self.verbose:
                print("Download complete.")
            
        if self.verbose:
            print(f"Reading {self.filename}...")
        self.vertices, self.faces, self.colors = read_obj(obj_path)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Retrieves clones of the loaded geometry tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - vertices: (V, 3) float32 coordinates.
                - faces: (F, 3) int64 face indices.
                - colors: (V, 3) float32 colors or None.
        """
        return (
            self.vertices.clone(),
            self.faces.clone(),
            self.colors.clone() if self.colors is not None else None
        )


class Alligator(Common3D):
    """A stylised alligator model, useful for testing extraction on elongated shapes with thin limbs.

    Downloads and caches `alligator.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Alligator
        >>> vertices, faces, colors = Alligator().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Alligator asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("alligator.obj", download_dir)


class Armadillo(Common3D):
    """The Stanford Armadillo, a standard benchmark with dense, high-curvature detail.

    Downloads and caches `armadillo.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Armadillo
        >>> vertices, faces, colors = Armadillo().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Armadillo asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("armadillo.obj", download_dir)


class Beast(Common3D):
    """A quadruped creature model with pronounced surface detail.

    Downloads and caches `beast.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Beast
        >>> vertices, faces, colors = Beast().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Beast asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("beast.obj", download_dir)


class BeetleAlt(Common3D):
    """An alternative beetle model variant with differing topology.

    Downloads and caches `beetle-alt.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import BeetleAlt
        >>> vertices, faces, colors = BeetleAlt().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the BeetleAlt asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("beetle-alt.obj", download_dir)


class Beetle(Common3D):
    """A beetle model exercising fine appendage geometry.

    Downloads and caches `beetle.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Beetle
        >>> vertices, faces, colors = Beetle().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Beetle asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("beetle.obj", download_dir)


class Bimba(Common3D):
    """A bust sculpture scan, a common target for smoothing and curvature tests.

    Downloads and caches `bimba.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Bimba
        >>> vertices, faces, colors = Bimba().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Bimba asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("bimba.obj", download_dir)


class Cheburashka(Common3D):
    """A cartoon character model with large smooth regions and sharp ear creases.

    Downloads and caches `cheburashka.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Cheburashka
        >>> vertices, faces, colors = Cheburashka().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Cheburashka asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("cheburashka.obj", download_dir)


class Cow(Common3D):
    """The classic cow model, widely used in parameterisation and remeshing literature.

    Downloads and caches `cow.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Cow
        >>> vertices, faces, colors = Cow().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Cow asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("cow.obj", download_dir)


class Fandisk(Common3D):
    """A CAD part with prominent sharp creases and mechanical corners, the standard test for feature-preserving extraction such as Dual Contouring.

    Downloads and caches `fandisk.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Fandisk
        >>> vertices, faces, colors = Fandisk().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Fandisk asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("fandisk.obj", download_dir)


class HappyBuddha(Common3D):
    """The Stanford Happy Buddha, a high-genus scan that stresses topological handling.

    Downloads and caches `happy.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import HappyBuddha
        >>> vertices, faces, colors = HappyBuddha().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the HappyBuddha asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("happy.obj", download_dir)


class Homer(Common3D):
    """A cartoon character model with mixed smooth and faceted regions.

    Downloads and caches `homer.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Homer
        >>> vertices, faces, colors = Homer().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Homer asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("homer.obj", download_dir)


class Horse(Common3D):
    """A horse model with thin legs, useful for testing narrow-band voxelisation.

    Downloads and caches `horse.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Horse
        >>> vertices, faces, colors = Horse().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Horse asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("horse.obj", download_dir)
        
        
class Igea(Common3D):
    """The Igea bust scan, a standard subject for detail-preserving simplification.

    Downloads and caches `igea.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Igea
        >>> vertices, faces, colors = Igea().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Igea asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("igea.obj", download_dir)


class Lucy(Common3D):
    """The Stanford Lucy statue, a very large scan suited to high-resolution extraction benchmarks.

    Downloads and caches `lucy.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Lucy
        >>> vertices, faces, colors = Lucy().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Lucy asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("lucy.obj", download_dir)


class MaxPlanck(Common3D):
    """The Max Planck bust, a common target for curvature and smoothing tests.

    Downloads and caches `max-planck.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import MaxPlanck
        >>> vertices, faces, colors = MaxPlanck().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the MaxPlanck asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("max-planck.obj", download_dir)


class Nefertiti(Common3D):
    """The Nefertiti bust scan, dense and highly detailed.

    Downloads and caches `nefertiti.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Nefertiti
        >>> vertices, faces, colors = Nefertiti().get()
    """

    def __init__(self, download_dir: str ="~/.conquer3d") -> None:
        """Downloads and caches the Nefertiti asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("nefertiti.obj", download_dir)


class Ogre(Common3D):
    """An ogre character model with heavy surface displacement.

    Downloads and caches `ogre.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Ogre
        >>> vertices, faces, colors = Ogre().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Ogre asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("ogre.obj", download_dir)


class RockerArm(Common3D):
    """A mechanical rocker arm, a CAD model combining curved and planar faces.

    Downloads and caches `rocker-arm.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import RockerArm
        >>> vertices, faces, colors = RockerArm().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the RockerArm asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("rocker-arm.obj", download_dir)


class Spot(Common3D):
    """Keenan Crane's Spot the cow, a genus-0 model widely used in geometry processing.

    Downloads and caches `spot.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Spot
        >>> vertices, faces, colors = Spot().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Spot asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("spot.obj", download_dir)


class StanfordBunny(Common3D):
    """The Stanford Bunny, the most widely used benchmark model in the field.

    Downloads and caches `stanford-bunny.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import StanfordBunny
        >>> vertices, faces, colors = StanfordBunny().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the StanfordBunny asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("stanford-bunny.obj", download_dir)
        
        
class Suzanne(Common3D):
    """Blender's Suzanne monkey head, a compact model with mixed topology.

    Downloads and caches `suzanne.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Suzanne
        >>> vertices, faces, colors = Suzanne().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Suzanne asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("suzanne.obj", download_dir)


class Teapot(Common3D):
    """The Utah Teapot, the canonical computer graphics test model.

    Downloads and caches `teapot.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Teapot
        >>> vertices, faces, colors = Teapot().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Teapot asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("teapot.obj", download_dir)


class Woody(Common3D):
    """A character model with articulated limbs.

    Downloads and caches `woody.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import Woody
        >>> vertices, faces, colors = Woody().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the Woody asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("woody.obj", download_dir)
        

class XYZRGBDragon(Common3D):
    """The XYZ RGB Dragon scan, a dense model with intricate scale detail.

    Downloads and caches `xyzrgb_dragon.obj` from the common-3d-test-models repository
    on first use. See :class:`Common3D` for the loading interface.

    Example:
        >>> from conquer3d.data.assets import XYZRGBDragon
        >>> vertices, faces, colors = XYZRGBDragon().get()
    """

    def __init__(self, download_dir: str = "~/.conquer3d") -> None:
        """Downloads and caches the XYZRGBDragon asset.

        Args:
            download_dir (str, optional): Local cache directory. Defaults to
                `"~/.conquer3d"`.
        """
        super().__init__("xyzrgb_dragon.obj", download_dir)
