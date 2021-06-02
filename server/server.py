import logging

import post_process
import run_cfd
from aiohttp import web
from visualise import VisualiseLevel


LOGGER = logging.getLogger(__name__)


async def hello_world(request: web.Request) -> web.Response:
    return web.Response(text="Hello world")


async def pointcloud(request: web.Request) -> web.Response:
    LOGGER.info("Incoming pointcloud")

    data = await request.json()
    points = data["points"]

    pcd = post_process.convert_to_pointcloud(points)
    post_process.save_pointcloud(pcd)

    upper_surface_coeffs, lower_surface_coeffs, blade_start, blade_end, aerofoil_mesh = post_process.post_process(
        pcd, visualise_level=VisualiseLevel.OFF,
    )

    return web.json_response(
        {
            "upper_coeffs": upper_surface_coeffs.tolist(),
            "lower_coeffs": lower_surface_coeffs.tolist(),
            "blade_start_x": blade_start[0],
            "blade_start_y": blade_start[1],
            "blade_end_x": blade_end[0],
            "blade_end_y": blade_end[1],
            "aerofoil_mesh": aerofoil_mesh,
        },
    )


async def cfd(request: web.Request) -> web.Response:
    LOGGER.info("Received CFD request")

    data = await request.json()
    upper_surface_coeffs = data["upper_coeffs"]
    lower_surface_coeffs = data["lower_coeffs"]

    (
        pitch,
        x,
        y,
        pstat_coeff,
        pstag_coeff,
        vx,
        vt,
        indices,
    ) = run_cfd.run_cfd(upper_surface_coeffs, lower_surface_coeffs)

    return web.json_response(
        {
            "pitch": pitch,
            "pstat_min": float(pstat_coeff.min()),
            "pstat_max": float(pstat_coeff.max()),
            "pstag_min": float(pstag_coeff.min()),
            "pstag_max": float(pstag_coeff.max()),
            "vx_min": float(vx.min()),
            "vx_max": float(vx.max()),
            "vt_min": float(vt.min()),
            "vt_max": float(vt.max()),
            "indices": indices,
            "points": [
                {
                    "x": float(x[i][j]),
                    "y": float(y[i][j]),
                    "i": i,
                    "j": j,
                    "pstat": float(pstat_coeff[i][j]),
                    "pstag": float(pstag_coeff[i][j]),
                    "vx": float(vx[i][j]),
                    "vt": float(vt[i][j]),
                }
                for i in range(x.shape[0])
                for j in range(x.shape[1])
            ],
        },
    )


app = web.Application(client_max_size=int(1e9))  # max ~1GB request body


app.add_routes(
    [
        web.get('/', hello_world),
        web.post('/pointcloud', pointcloud),
        web.post('/cfd', cfd),
    ],
)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)  # silence matplotlib debug messages

    web.run_app(app)
