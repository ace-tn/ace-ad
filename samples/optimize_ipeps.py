import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from acead.ipeps import Ipeps

def optimize_ipeps(ipeps_config, init_tensor=None):
    ipeps = Ipeps(ipeps_config, init_tensor)
    lbfgs_steps = ipeps_config.get('lbfgs_steps')

    def closure():
        optimizer.zero_grad()
        ene = ipeps()[0]
        ene.backward()
        return ene

    optimizer = torch.optim.AdamW(ipeps.parameters())

    ene_old = 0
    for i in range(1, 10000+1):
        ene = optimizer.step(closure)
        if i % 1 == 0:
            with torch.inference_mode():
                ene_new, mag_x, mag_y, mag_z = ipeps()
                ene_diff = ene_new-ene_old
                mag = torch.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
                message = ('{} ' + 6*'{:.8f} ').format(i, ene_new, ene_new-ene_old, mag_x, mag_y, mag_z, mag)
                print ('iter, ene, ene diff, mag_x, mag_y, mag_z, mag', message)
                ene_old = ene_new
                if abs(ene_diff) < 1e-6 and i > 20:
                    break

    optimizer = torch.optim.SGD(ipeps.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    ene_old = 0
    for i in range(1, 10000+1):
        ene = optimizer.step(closure)
        scheduler.step(ene)
        if i % 1 == 0:
            with torch.inference_mode():
                ene_new, mag_x, mag_y, mag_z = ipeps()
                ene_diff = ene_new-ene_old
                mag = torch.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
                message = ('{} ' + 6*'{:.8f} ').format(i, ene_new, ene_new-ene_old, mag_x, mag_y, mag_z, mag)
                print ('iter, ene, ene diff, mag_x, mag_y, mag_z, mag', message)
                ene_old = ene_new
                if abs(ene_diff) < 1e-6 and i > 20:
                    break


    optimizer = torch.optim.LBFGS(
        ipeps.parameters(),
        lr=1e-2,
        max_iter=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-8,
    )

    for i in range(1, lbfgs_steps+1):
        optimizer.step(closure)
        if i % 1 == 0:
            with torch.inference_mode():
                ene_new, mag_x, mag_y, mag_z = ipeps()
                mag = torch.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
                message = ('{} ' + 6*'{:.8f} ').format(i, ene_new, ene_new-ene_old, mag_x, mag_y, mag_z, mag)
                print ('iter, ene, ene diff, mag_x, mag_y, mag_z, mag', message)
                ene_old = ene_new

    mag = torch.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    measurements = {
        "ene": ene_new,
        "mag_x": mag_x,
        "mag_y": mag_y,
        "mag_z": mag_z,
        "mag": mag,
    }

    return ipeps, measurements


if __name__=='__main__':
    dims = {}
    dims['phys'] = 2
    dims['bond'] = 2
    dims['chi'] = 20

    ctmrg_steps = 5
    lbfgs_steps = 0

    ipeps_config = {
        'dtype': torch.float64,
        'device': torch.device('cpu'),
        'dims': dims,
        'ctmrg_steps': ctmrg_steps,
        'lbfgs_steps': lbfgs_steps,
        'model':{
            'name': 'ising',
            'params':{
                'jz': 1.0,
                'hx': 2.05,
            }
        },
    }

    precomputed_measurements = {
        "D": 2,
        "chi": 20,
        "hx": 2.05,
        "ene": -2.53785850,
        "mag_x": 0.53835030,
        "mag_z": 0.83657445,
        "mag": 0.99482554,
    }

    _,measurements = optimize_ipeps(ipeps_config)

    print(f"\nmeasurement differences for hx={ipeps_config['model']['params']['hx']}:\n"
          + f"energy: {abs(measurements['ene'] - precomputed_measurements['ene'])}\n"
          + f"mag_x: {abs(abs(measurements['mag_x']) - precomputed_measurements['mag_x'])}\n"
          + f"mag_z: {abs(abs(measurements['mag_z']) - precomputed_measurements['mag_z'])}\n"
    )
