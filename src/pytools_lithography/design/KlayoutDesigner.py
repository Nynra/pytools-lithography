import matplotlib
from matplotlib.patches import Rectangle, Circle
from enum import Enum
from .pya import *
import numpy as np
import os


def get_template_path(filename:str) -> str:
    """Get the path of the template file.

    Parameters
    ----------
    filename : str
        The name of the template file

    Returns
    -------
    str
        The absolute path of the template file
    """
    path = os.path.join(os.path.dirname(__file__), "templates", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template file {filename} not found")
    return path


class ORIENTATION(Enum):
    """Enum class for the ised for the orientation of the vernier scale."""
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3


COLORS = ["#fcba03", "#a63019", "#1c67b8"]


class Designer:
    def __init__(
        self,
        layout,
        dbu,
        layer_amount: int,
        ax,
        temp_filename,
        marker_filename,
        output_filename,
    ) -> None:

        self.temp_filename = temp_filename
        self.marker_filename = marker_filename
        self.output_filename = output_filename

        self.kdrawer = KlayoutDrawer(
            layout, dbu, layer_amount, temp_filename, marker_filename, output_filename
        )
        self.ndrawer = NotebookDrawer(ax)

    def vernier(
        self,
        x: int,
        y: int,
        ms_spacing: float,
        vs_spacing: float,
        WIDTH,
        ms_layerIdx: int,
        vs_layerIdx: int,
        max_length: int,
        orientation: ORIENTATION,
    ):
        self.kdrawer.construct_vernier(
            x,
            y,
            ms_spacing,
            vs_spacing,
            WIDTH,
            ms_layerIdx,
            vs_layerIdx,
            max_length,
            orientation,
        )
        self.ndrawer.render_vernier(
            x,
            y,
            ms_spacing,
            vs_spacing,
            WIDTH,
            ms_layerIdx,
            vs_layerIdx,
            max_length,
            orientation,
        )

    def bar(self, x: int, y: int, width, height, layer):
        self.kdrawer.construct_bar(x, y, width, height, layer)
        self.ndrawer.render_bar(x, y, width, height, layer)

    def cross(self, x: int, y: int, width, height, layer):
        self.bar(x, y, width, height, layer)
        self.bar(x, y, height, width, layer)

    def marker(self, layer, *positions):
        if self.marker_filename == "Default_Marker.gds":
            print(f"No marker selected! Default marker used!")
        else:
            print(
                f"Marker selected! GDS will contain: {self.marker_filename[:-4]} as a marker."
            )
        for pos in positions:
            self.ndrawer.render_marker(pos[0], pos[1], layer)
            self.kdrawer.construct_marker(*positions)

    def draw(self):
        self.kdrawer.draw()


class NotebookDrawer(Designer):
    def __init__(self, ax : matplotlib.axes.Axes) -> ...:
        """Initializes the NotebookDrawer with a given matplotlib axis."""
        self.ax = ax
        self.ax.grid()
        self.ax.set_xlim([-1700, 1700])
        self.ax.set_ylim([-700, 700])

    def render_vernier(
        self,
        x: int,
        y: int,
        ms_spacing: float,
        vs_spacing: float,
        WIDTH : int,
        ms_layerIdx: int,
        vs_layerIdx: int,
        max_length: int,
        orientation: ORIENTATION,
    ) -> ...:
        """Render a vernier scale based on given coordinates, sizes and orientation in a mpl graph.

        Parameters
        ----------
        x : int
            X position of the mid point of the vernier scale
        y : int
            Y position of the mid point of the vernier scale
        ms_spacing : float
            The spacing between the ticks on the main scale
        vs_spacing : float
            The spacing between the ticks on the second scale
        WIDTH : int
            The width of the ticks
        ms_layerIdx : int
            The layer index of the main scale
        vs_layerIdx : int
            The layer index of the vernier scale
        max_length : int
            The maximum length the scale is allowed to be
        """
        # WIDTH = 10 # micrometer
        MID_TICK_HEIGHT = 50  # micrometer

        mid_tick_orientation_coords = {
            ORIENTATION.TOP: [
                ((x, y), WIDTH, MID_TICK_HEIGHT),
                ((x, y), WIDTH, -MID_TICK_HEIGHT),
            ],
            ORIENTATION.BOTTOM: [
                ((x, y), WIDTH, -MID_TICK_HEIGHT),
                ((x, y), WIDTH, MID_TICK_HEIGHT),
            ],
            ORIENTATION.LEFT: [
                ((x, y), -MID_TICK_HEIGHT, WIDTH),
                ((x, y), MID_TICK_HEIGHT, WIDTH),
            ],
            ORIENTATION.RIGHT: [
                ((x, y), MID_TICK_HEIGHT, WIDTH),
                ((x, y), -MID_TICK_HEIGHT, WIDTH),
            ],
        }

        # Assigning coordinates for main and vernier scale based on orientation
        ms_coords, vs_coords = mid_tick_orientation_coords[orientation]

        self.ax.add_patch(Rectangle(*ms_coords, color=COLORS[ms_layerIdx]))
        self.ax.add_patch(Rectangle(*vs_coords, color=COLORS[vs_layerIdx]))

        number_of_stalks = max_length / (WIDTH + ms_spacing)
        for n in range(1, int(np.ceil(number_of_stalks / 2))):
            height = 40 if n % 10 == 0 else 25  # height in micrometers

            reg_ticks_orientation_coords = {
                ORIENTATION.TOP: {
                    "ms_p_tick": ((x + n * (WIDTH + ms_spacing), y), WIDTH, height),
                    "ms_n_tick": (
                        (x + WIDTH / 2 - n * (WIDTH + ms_spacing), y),
                        WIDTH,
                        height,
                    ),
                    "vs_p_tick": ((x + n * (WIDTH + vs_spacing), y), WIDTH, -height),
                    "vs_n_tick": (
                        (x + WIDTH / 2 - n * (WIDTH + vs_spacing), y),
                        WIDTH,
                        -height,
                    ),
                },
                ORIENTATION.BOTTOM: {
                    "ms_p_tick": ((x + n * (WIDTH + ms_spacing), y), WIDTH, -height),
                    "ms_n_tick": (
                        (x + WIDTH / 2 - n * (WIDTH + ms_spacing), y),
                        WIDTH,
                        -height,
                    ),
                    "vs_p_tick": ((x + n * (WIDTH + vs_spacing), y), WIDTH, height),
                    "vs_n_tick": (
                        (x + WIDTH / 2 - n * (WIDTH + vs_spacing), y),
                        WIDTH,
                        height,
                    ),
                },
                ORIENTATION.LEFT: {
                    "ms_p_tick": (
                        (x, y + n * (WIDTH + ms_spacing) - WIDTH / 2),
                        -height,
                        WIDTH,
                    ),
                    "ms_n_tick": (
                        (x, y + WIDTH / 2 - n * (WIDTH + ms_spacing)),
                        -height,
                        WIDTH,
                    ),
                    "vs_p_tick": (
                        (x, y + n * (WIDTH + vs_spacing) - WIDTH / 2),
                        height,
                        WIDTH,
                    ),
                    "vs_n_tick": (
                        (x, y + WIDTH / 2 - n * (WIDTH + vs_spacing)),
                        height,
                        WIDTH,
                    ),
                },
                ORIENTATION.RIGHT: {
                    "ms_p_tick": (
                        (x, y + n * (WIDTH + ms_spacing) - WIDTH / 2),
                        height,
                        WIDTH,
                    ),
                    "ms_n_tick": (
                        (x, y + WIDTH / 2 - n * (WIDTH + ms_spacing)),
                        height,
                        WIDTH,
                    ),
                    "vs_p_tick": (
                        (x, y + n * (WIDTH + vs_spacing) - WIDTH / 2),
                        -height,
                        WIDTH,
                    ),
                    "vs_n_tick": (
                        (x, y + WIDTH / 2 - n * (WIDTH + vs_spacing)),
                        -height,
                        WIDTH,
                    ),
                },
            }
            ms_p_tick, ms_n_tick, vs_p_tick, vs_n_tick = reg_ticks_orientation_coords[
                orientation
            ].values()
            self.ax.add_patch(Rectangle(*ms_p_tick, color=COLORS[ms_layerIdx]))
            self.ax.add_patch(Rectangle(*ms_n_tick, color=COLORS[ms_layerIdx]))
            self.ax.add_patch(Rectangle(*vs_p_tick, color=COLORS[vs_layerIdx]))
            self.ax.add_patch(Rectangle(*vs_n_tick, color=COLORS[vs_layerIdx]))

    def render_bar(self, x: int, y: int, width: int, height: int, layerIdx: int) -> ...:
        """Render a bar based on given coordinates, sizes and layer index in a mpl graph.

        Parameters
        ----------
        x : int
            X position of the mid point of the bar
        y : int
            Y position of the mid point of the bar
        width : int
            The width of the bar
        height : int
            The height of the bar
        layerIdx : int
            The layer index of the bar
        """
        self.ax.add_patch(
            Rectangle(
                (x - width / 2, y - height / 2), width, height, color=COLORS[layerIdx]
            )
        )

    def render_marker(self, x: int, y: int, layerIdx: int) -> ...:
        """Render a marker based on given coordinates and layer index in a mpl graph.

        Parameters
        ----------
        x : int
            X position of the mid point of the marker
        y : int
            Y position of the mid point of the marker
        layerIdx : int
            The layer index of the marker
        """
        self.ax.add_patch(Circle((x, y), 25, color=COLORS[layerIdx], fill=False))
        self.render_bar(x, y, 2, 100, layerIdx)
        self.render_bar(x, y, 100, 2, layerIdx)


class KlayoutDrawer(Designer):

    def __init__(
        self,
        layout: Layout,
        dbu: float,
        layer_amount: int,
        temp_filename: str,
        marker_filename: str,
        output_filename: str,
    ) -> ...:

        self.temp_filename = temp_filename
        self.marker_filename = marker_filename
        self.output_filename = output_filename

        self.layout = layout
        self.layout.dbu = dbu
        self.top = self.layout.create_cell("TopCell")
        self.layers = self.construct_layers(layer_amount)

    def construct_layers(self, layer_amount: int) -> ...:
        """Constructs the layers based on the amount of layers given.

        Parameters
        ----------
        layer_amount : int
            The amount of layers to be created
        """
        layer_array = []
        for i in range(layer_amount):
            layer_array.append(self.layout.layer(i, 0))
        return layer_array

    def construct_bar(self, x: int, y: int, width: int, height: int, layerIdx: int) -> ...:
        """Construct a bar based on given coordinates, sizes and layer index.

        Parameters
        ----------
        x : int
            X position of the mid point of the bar
        y : int
            Y position of the mid point of the bar
        width : int
            The width of the bar
        height : int
            The height of the bar
        layerIdx : int
            The layer index of the bar
        """
        box = DBox(x - width / 2, y + height / 2, x + width / 2, y - height / 2)
        self.top.shapes(self.layers[layerIdx]).insert(box)

    def construct_vernier(
        self,
        x: int,
        y: int,
        ms_spacing: float,
        vs_spacing: float,
        WIDTH: int,
        ms_layerIdx: int,
        vs_layerIdx: int,
        max_length: int,
        orientation: ORIENTATION,
    ):
        """Construct a vernier scale based on given coordinates, sizes and orientation.

        Parameters
        ----------
        x : int
            X position of the mid point of the vernier scale
        y : int
            Y position of the mid point of the vernier scale
        ms_spacing : float
            The spacing between the ticks on the main scale
        vs_spacing : float
            The spacing between the ticks on the second scale
        WIDTH : int
            The width of the ticks
        ms_layerIdx : int
            The layer index of the main scale
        vs_layerIdx : int
            The layer index of the vernier scale
        max_length : int
            The maximum length the scale is allowed to be
        orientation : ORIENTATION
            The orientation of the scale
        """
        # WIDTH = 10 # micrometer
        MID_TICK_HEIGHT = 50  # micrometer

        mid_tick_orientation_coords = {
            ORIENTATION.TOP: [
                (x - WIDTH / 2, y, x + WIDTH / 2, y + MID_TICK_HEIGHT),
                (x - WIDTH / 2, y - MID_TICK_HEIGHT, x + WIDTH / 2, y),
            ],
            ORIENTATION.BOTTOM: [
                (x - WIDTH / 2, y - MID_TICK_HEIGHT, x + WIDTH / 2, y),
                (x - WIDTH / 2, y, x + WIDTH / 2, y + MID_TICK_HEIGHT),
            ],
            ORIENTATION.LEFT: [
                (x, y - WIDTH / 2, x - MID_TICK_HEIGHT, y + WIDTH / 2),
                (x + MID_TICK_HEIGHT, y - WIDTH / 2, x, y + WIDTH / 2),
            ],
            ORIENTATION.RIGHT: [
                (x + MID_TICK_HEIGHT, y - WIDTH / 2, x, y + WIDTH / 2),
                (x, y - WIDTH / 2, x - MID_TICK_HEIGHT, y + WIDTH / 2),
            ],
        }

        # Assigning coordinates for main and vernier scale based on orientation
        ms_coords, vs_coords = mid_tick_orientation_coords[orientation]

        # Make the centerline for the mainscale
        ms_main_tick = DBox(*ms_coords)

        # Make the centerline for the vernierscale
        vs_main_tick = DBox(*vs_coords)

        # Add shape to its respective layer
        self.top.shapes(self.layers[ms_layerIdx]).insert(ms_main_tick)
        self.top.shapes(self.layers[vs_layerIdx]).insert(vs_main_tick)

        number_of_ticks = max_length / (WIDTH + ms_spacing)
        for n in range(1, int(np.ceil(number_of_ticks / 2))):
            height = 40 if n % 10 == 0 else 25  # height in micrometers

            reg_ticks_orientation_coords = {
                ORIENTATION.TOP: {
                    "ms_p_tick": (
                        x + n * (WIDTH + ms_spacing) - WIDTH / 2,
                        y,
                        x + ((n + 1) * WIDTH) + n * ms_spacing - WIDTH / 2,
                        y + height,
                    ),
                    "ms_n_tick": (
                        x + WIDTH / 2 - n * (WIDTH + ms_spacing),
                        y,
                        x + WIDTH / 2 - ((n + 1) * WIDTH) - n * ms_spacing,
                        y + height,
                    ),
                    "vs_p_tick": (
                        x + n * (WIDTH + vs_spacing) - WIDTH / 2,
                        y - height,
                        x + ((n + 1) * WIDTH) + n * vs_spacing - WIDTH / 2,
                        y,
                    ),
                    "vs_n_tick": (
                        x + WIDTH / 2 - n * (WIDTH + vs_spacing),
                        y - height,
                        x + WIDTH / 2 - ((n + 1) * WIDTH) - n * vs_spacing,
                        y,
                    ),
                },
                ORIENTATION.BOTTOM: {
                    "ms_p_tick": (
                        x + n * (WIDTH + ms_spacing) - WIDTH / 2,
                        y - height,
                        x + ((n + 1) * WIDTH) + n * ms_spacing - WIDTH / 2,
                        y,
                    ),
                    "ms_n_tick": (
                        x + WIDTH / 2 - n * (WIDTH + ms_spacing),
                        y - height,
                        x + WIDTH / 2 - ((n + 1) * WIDTH) - n * ms_spacing,
                        y,
                    ),
                    "vs_p_tick": (
                        x + n * (WIDTH + vs_spacing) - WIDTH / 2,
                        y,
                        x + ((n + 1) * WIDTH) + n * vs_spacing - WIDTH / 2,
                        y + height,
                    ),
                    "vs_n_tick": (
                        x + WIDTH / 2 - n * (WIDTH + vs_spacing),
                        y,
                        x + WIDTH / 2 - ((n + 1) * WIDTH) - n * vs_spacing,
                        y + height,
                    ),
                },
                ORIENTATION.RIGHT: {
                    "ms_p_tick": (
                        x + height,
                        y + n * (WIDTH + ms_spacing) - WIDTH / 2,
                        x,
                        y + ((n + 1) * WIDTH) + n * ms_spacing - WIDTH / 2,
                    ),
                    "ms_n_tick": (
                        x + height,
                        y + WIDTH / 2 - n * (WIDTH + ms_spacing),
                        x,
                        y + WIDTH / 2 - ((n + 1) * WIDTH) - n * ms_spacing,
                    ),
                    "vs_p_tick": (
                        x,
                        y + n * (WIDTH + vs_spacing) - WIDTH / 2,
                        x - height,
                        y + ((n + 1) * WIDTH) + n * vs_spacing - WIDTH / 2,
                    ),
                    "vs_n_tick": (
                        x,
                        y + WIDTH / 2 - n * (WIDTH + vs_spacing),
                        x - height,
                        y + WIDTH / 2 - ((n + 1) * WIDTH) - n * vs_spacing,
                    ),
                },
                ORIENTATION.LEFT: {
                    "ms_p_tick": (
                        x,
                        y + n * (WIDTH + ms_spacing) - WIDTH / 2,
                        x - height,
                        y + ((n + 1) * WIDTH) + n * ms_spacing - WIDTH / 2,
                    ),
                    "ms_n_tick": (
                        x,
                        y + WIDTH / 2 - n * (WIDTH + ms_spacing),
                        x - height,
                        y + WIDTH / 2 - ((n + 1) * WIDTH) - n * ms_spacing,
                    ),
                    "vs_p_tick": (
                        x + height,
                        y + n * (WIDTH + vs_spacing) - WIDTH / 2,
                        x,
                        y + ((n + 1) * WIDTH) + n * vs_spacing - WIDTH / 2,
                    ),
                    "vs_n_tick": (
                        x + height,
                        y + WIDTH / 2 - n * (WIDTH + vs_spacing),
                        x,
                        y + WIDTH / 2 - ((n + 1) * WIDTH) - n * vs_spacing,
                    ),
                },
            }

            # Assigning coordinates for main and vernier scale based (negative an positive direction) on orientation
            ms_p_tick, ms_n_tick, vs_p_tick, vs_n_tick = reg_ticks_orientation_coords[
                orientation
            ].values()

            # Make all boxes in negative an positive direction
            ms_p_tick = DBox(*ms_p_tick)
            ms_n_tick = DBox(*ms_n_tick)
            vs_p_tick = DBox(*vs_p_tick)
            vs_n_tick = DBox(*vs_n_tick)

            # Add shape to its respective layer
            self.top.shapes(self.layers[ms_layerIdx]).insert(ms_p_tick)
            self.top.shapes(self.layers[ms_layerIdx]).insert(ms_n_tick)
            self.top.shapes(self.layers[vs_layerIdx]).insert(vs_p_tick)
            self.top.shapes(self.layers[vs_layerIdx]).insert(vs_n_tick)

    def construct_marker(self, *positions) -> ...: 
        """Construct a marker based on given coordinates in the layout.

        Parameters
        ----------
        positions : tuple
            The positions of the markers
        """
        layout = Layout()
        top_cell = layout.create_cell("TOP")

        x = 0
        y = 0
        files = [self.temp_filename] + [self.marker_filename] * len(positions)
        for idx, file in enumerate(files):
            layout_import = Layout()
            layout_import.read(file)

            imported_top_cell = layout_import.top_cell()
            target_cell = layout.create_cell(imported_top_cell.name)
            target_cell.copy_tree(imported_top_cell)

            layout_import._destroy()

            inst = DCellInstArray(
                target_cell.cell_index(), DTrans(DVector(x, y))
            )
            top_cell.insert(inst)

            x = positions[idx - 1][0]
            y = positions[idx - 1][1]
        layout.write(self.output_filename)

    def draw(self) -> ...:
        """Draw the layout.
        
        The layout will be drawn and saved to the temp_filename set using
        the constructor of the class.
        """
        self.layout.write(self.temp_filename)
