import sys, os
BASE_PATH = os.path.dirname("__file__")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import json
from planning_project.env.env import GridMap
from planning_project.env.slip_models import SlipModel, SlipModelsGenerator
from planning_project.env.utils import SlipDistributionMap
from planning_project.planner.planner import AStarPlanner
from planning_project.utils.data import DataSet, create_int_label
from scripts.eval import PlanMetrics
import cProfile
import pstats
import argparse
import plotly.io as pio
import traceback


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run path planning with user-defined start and goal positions.")
parser.add_argument("--start_x", type=float, required=True, help="X-coordinate of the start position")
parser.add_argument("--start_y", type=float, required=True, help="Y-coordinate of the start position")
parser.add_argument("--goal_x", type=float, required=True, help="X-coordinate of the goal position")
parser.add_argument("--goal_y", type=float, required=True, help="Y-coordinate of the goal position")

args = parser.parse_args()



n_terrains = 10
res = 1
start_pos = (args.start_x, args.start_y)
goal_pos = (args.goal_x, args.goal_y)
idx_instance = 0
type_model = 'gtm'
type_embed = "cvar"
if type_embed == "mean":
    alpha = None
else:
    alpha = 0.99


# set directory paths to classifiers and testing data
data_dir = os.path.join(
    BASE_PATH,
    'C:/Users/chakr/Desktop/Planetary_Terrains'
)


dataset = DataSet(data_dir, "predicted_npy")
smg = SlipModelsGenerator(
    dirname=data_dir,
    n_terrains=n_terrains
)
planner = AStarPlanner(
    map=None,
    smg=smg,
    nn_model_dir=None
)
plan_metrics = PlanMetrics(
    is_plan=True,
    type_model=type_model,
    type_embed=type_embed,
    alpha=alpha
)


# set map instance for planning
color, mask = dataset[idx_instance]
color = dataset.to_image(color)
mask = create_int_label(dataset.to_image(mask))
height = dataset.get_height_map(idx_instance)

grid_map = GridMap(color.shape[1], res)
grid_map.load_env(height, mask, color)
sdm = SlipDistributionMap(grid_map, smg)
grid_map.data.slip = sdm.set_slip()


print("Height Map generated")
plt.figure(figsize=(6, 6))
plt.imshow(height, cmap="terrain")  # Use terrain colormap
plt.colorbar(label="Height")  # Add a color bar for reference
plt.title("Height Map")
plt.axis("off")  # Hide axes
plt.show()





def profile_search():
    print("Path planning has started")
    profiler = cProfile.Profile()
    profiler.enable()
    # Run your search
    planner.search_path()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()



planner.reset(
    map=grid_map,
    start_pos=start_pos,
    goal_pos=goal_pos,
    plan_metrics=plan_metrics
)

profile_search()

metrics = planner.execute_final_path()



fig, ax = planner.plot_envs(figsize=(18, 8), is_tf = False)
planner.plot_final_path(
    ax = ax,
    metrics=metrics,
    color="red",
    plan_type="%s with %s" % (plan_metrics.type_model, plan_metrics.type_embed)
)




def plot_3d_path_map(self, metrics, figsize=(10, 8), is_tf=True, color="red"):
    """
    Create a separate 3D visualization of the terrain with path

    :param metrics: metrics containing planning results
    :param figsize: size of the figure
    :param is_tf: existence of terrain features
    :param color: color of the path
    :return: figure and axis objects
    """
    # Create a new figure for 3D visualization
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Create terrain visualization using viz_3d_map
    xx, yy = np.meshgrid(
        np.arange(0.0, self.map.n * self.map.res, self.map.res),
        np.arange(0.0, self.map.n * self.map.res, self.map.res)
    )

    grid_data = np.reshape(self.map.data.height, (self.map.n, self.map.n))
    data = self.map.data.height

    # Plot terrain
    if not is_tf:
        hmap = ax.plot_surface(
            xx + self.map.res / 2.0,
            yy + self.map.res / 2.0,
            grid_data,
            cmap="viridis",
            vmin=min(data),
            vmax=max(data),
            linewidth=0,
            antialiased=True,
            alpha=0.8  # Make terrain slightly transparent
        )
    else:
        hmap = ax.plot_surface(
            xx + self.map.res / 2.0,
            yy + self.map.res / 2.0,
            grid_data,
            facecolors=self.map.data.color,
            linewidth=0,
            antialiased=True,
            alpha=0.8  # Make terrain slightly transparent
        )

    # Plot path if available
    if metrics.path is not None and len(metrics.path) > 0:
        # Extract coordinates
        x_coords = metrics.path[:, 0]
        y_coords = metrics.path[:, 1]
        z_coords = metrics.path[:, 2] + 0.1  # Small offset for visibility

        # Plot the path
        line = ax.plot3D(
            x_coords, y_coords, z_coords,
            linewidth=3,
            color=color,
            label=f"Path (cost: {metrics.est_cost/60:.2f} min)"
        )

        # Plot start point
        start_pos = self.calc_pos_from_xy_id(self.node_start.xy_ids)
        ax.scatter(
            start_pos[0], start_pos[1], start_pos[2] + 0.1,
            marker="s",
            s=200,
            color="blue",
            edgecolor="black",
            label="Start"
        )

        # Plot goal point
        goal_pos = self.calc_pos_from_xy_id(self.node_goal.xy_ids)
        ax.scatter(
            goal_pos[0], goal_pos[1], goal_pos[2] + 0.1,
            marker="*",
            s=300,
            color="yellow",
            edgecolor="black",
            label="Goal"
        )

        # Plot failure point if exists
        if metrics.node_failed is not None:
            failed_pos = self.calc_pos_from_xy_id(metrics.node_failed.xy_ids)
            ax.scatter(
                failed_pos[0], failed_pos[1], failed_pos[2] + 0.1,
                marker="X",
                s=250,
                color="red",
                edgecolor="black",
                label="Failure"
            )

    # Set labels and adjust view
    ax.set_xlabel("x-axis (m)")
    ax.set_ylabel("y-axis (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("3D Terrain with Planned Path")

    # Adjust view angle and aspect ratio
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect((1, 1, 0.2))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_zticks([])  # Remove z-axis ticks


    # Set axis limits
    ax.set_xlim(xx.min(), xx.max() + self.map.res)
    ax.set_ylim(yy.min(), yy.max() + self.map.res)
    ax.set_zlim(min(data), xx.max() / 10)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    return fig, ax




import plotly.graph_objects as go
import numpy as np

def plot_3d_path_map_interactive(self, metrics, is_tf=True, path_color='yellow', output_file="static/terrain.html"):
    """
    Create an interactive 3D terrain visualization with the planned path.
    """
    xx, yy = np.meshgrid(
        np.arange(0.0, self.map.n * self.map.res, self.map.res),
        np.arange(0.0, self.map.n * self.map.res, self.map.res)
    )
    grid_data = np.reshape(self.map.data.height, (self.map.n, self.map.n))

    fig = go.Figure()

    if is_tf:
        lunar_texture = self.map.data.color
        if lunar_texture.ndim == 3 and lunar_texture.shape[-1] in [3, 4]:
            lunar_texture = np.dot(lunar_texture[..., :3], [0.2989, 0.5870, 0.1140])
        lunar_texture = lunar_texture / lunar_texture.max()

        fig.add_trace(go.Surface(
            x=xx + self.map.res / 2.0,
            y=yy + self.map.res / 2.0,
            z=grid_data,
            surfacecolor=lunar_texture,
            colorscale="gray",
            showscale=False,
            opacity=1.0
        ))
    else:
        fig.add_trace(go.Surface(
            x=xx + self.map.res / 2.0,
            y=yy + self.map.res / 2.0,
            z=grid_data,
            colorscale='earth',
            showscale=True,
            opacity=0.8
        ))

    if metrics.path is not None and len(metrics.path) > 0:
        x_coords = metrics.path[:, 0]
        y_coords = metrics.path[:, 1]
        z_coords = metrics.path[:, 2] + 0.15

        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            line=dict(color=path_color, width=6),
            marker=dict(size=4, color=path_color),
            name="Planned Path"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="x-axis (m)",
            yaxis_title="y-axis (m)",
            zaxis_title="Height (m)",
            aspectratio=dict(x=1, y=1, z=0.2),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        title="Interactive 3D Lunar Terrain with Path",
        showlegend=True
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pio.write_html(fig, output_file)
    return output_file


plot_3d_path_map(planner, metrics, figsize=(10, 8), is_tf=False, color = 'black')


fig = plot_3d_path_map_interactive(
    planner,
    metrics=metrics,
    is_tf = True,
    path_color="yellow",
)


if __name__ == '__main__':
    try:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--start_x', type=float, required=True)
        parser.add_argument('--start_y', type=float, required=True)
        parser.add_argument('--goal_x', type=float, required=True)
        parser.add_argument('--goal_y', type=float, required=True)
        args = parser.parse_args()

        # Initialize planner
        planner = AStarPlanner(
            map=grid_map,
            smg=smg,
            nn_model_dir=None
        )
        
        # Reset planner with positions
        planner.reset(
            map=grid_map,
            start_pos=(args.start_x, args.start_y),
            goal_pos=(args.goal_x, args.goal_y),
            plan_metrics=plan_metrics
        )

        # Search for path
        planner.search_path()
        metrics = planner.execute_final_path()

        if metrics and metrics.path is not None:
            # Generate visualization
            html_path = plot_3d_path_map_interactive(planner, metrics)
            print(f"VISUALIZATION_PATH:{html_path}")

            # Output path coordinates
            path_coords = []
            for point in metrics.path:
                # Use the z-coordinate directly from the path point
                path_coords.append({
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'z': float(point[2])  # The height is already included in metrics.path
                })
            print(f"PATH_COORDS:{json.dumps(path_coords)}")
        else:
            print("ERROR:No path found", file=sys.stderr)

    except Exception as e:
        print(f"ERROR:{str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) 