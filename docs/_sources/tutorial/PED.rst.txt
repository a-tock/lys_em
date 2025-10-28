Simulation of Precession Electron Diffraction (PED) Pattern
=================================================================

Here, we simulate a precession electron diffraction (PED) pattern using the multislice method based on the crystal structure of gold (Au).

**1. Creating a Crystal Structure**

Create the crystal structure of gold using lys_mat.CrystalStructure::

    from lys_mat import CrystalStructure
    crys = CrystalStructure.loadFrom("data/Au.cif")
    print(crys)
    # Symmetry: cubic Fm-3m (No. 225), Point group: m-3m
    # a = 4.07825, b = 4.07825, c = 4.07825, alpha = 90.00000, beta = 90.00000, gamma = 90.00000
    # --- atoms (4) ---
    # 1: Au (Z = 79, Occupancy = 1) Pos = (0.00000, 0.00000, 0.00000)
    # 2: Au (Z = 79, Occupancy = 1) Pos = (0.00000, 0.50000, 0.50000)
    # 3: Au (Z = 79, Occupancy = 1) Pos = (0.50000, 0.00000, 0.50000)
    # 4: Au (Z = 79, Occupancy = 1) Pos = (0.50000, 0.50000, 0.00000)

**2. Defining the Computational Space**

Define the simulation space using the FunctionSpace class::

    from lys_em import FunctionSpace
    sp = FunctionSpace.fromCrystal(crys, 128, 128, 50)

**3. Creating the Crystal Potential**

Generate the potential field formed by the crystal using the CrystalPotential class::

    from lys_em import CrystalPotential
    pot = CrystalPotential(sp, crys)

**4. Defining the Electron Microscope Parameters**

Set up the electron microscope parameters using the TEM class.
The tilt of the incident electron beam is defined using the TEMParameter class and passed as a list to the optional argument params::

    import numpy as np
    from lys_em import TEM, TEMParameter
    tem = TEM(60e3, params=[TEMParameter(tilt=[2, phi]) for phi in np.arange(0, 360, 360 / 90)])
    # In TEMParameter.tilt, tilt[0] is the beam tilt angle from the optical axis (in degrees),
    # and tilt[1] is the rotation angle within the plane perpendicular to the optical axis.

**5. Running the Multislice Simulation**

Run the multislice calculation using the setup from steps 1–4.
The data are returned as a 3D array, where the 0th dimension corresponds to the precession rotation angle.
By summing over this dimension, we obtain the simulated precession diffraction pattern::

    from lys_em import multislice, diffraction
    data = diffraction(multislice(pot, tem)).sum(axis=0)

Visualize the simulated diffraction pattern using matplotlib::

    import matplotlib.pyplot as plt
    plt.imshow(data, vmax=1e7)
    plt.show()

.. image:: ./image_PED/Au_PED.png

**6. Optional: Rearranging the Data**

As in the SAD simulation, rearranging the data improves the visual appearance, making it similar to experimentally obtained diffraction patterns::

    def rearrangeDiffraction(data, size, interval):
        res = np.zeros((size, size))
        for i in range(-size//(interval*2), size//(interval*2)):
            for j in range(-size//(interval*2), size//(interval*2)):
                res[i*interval, j*interval] = data[i, j]
        res = np.roll(res, (int(size/2), int(size/2)), axis=(0, 1))
        return res

    data = rearrangeDiffraction(data, 201, 5)
    plt.imshow(data, vmax=1e7)
    plt.show()

.. image:: ./image_PED/Au_PED_rearrange.png

**Summary**

Below is the complete code up to the execution of the multislice simulation::

    import numpy as np
    from lys_mat import CrystalStructure
    from lys_em import FunctionSpace, CrystalPotential, TEM, TEMParameter, multislice, diffraction

    crys = CrystalStructure.loadFrom("data/Au.cif")
    sp = FunctionSpace.fromCrystal(crys, 128, 128, 50)
    pot = CrystalPotential(sp, crys)
    tem = TEM(60e3, params=[TEMParameter(tilt=[2, phi]) for phi in np.arange(0, 360, 360 / 90)])
    data = diffraction(multislice(pot, tem)).sum(axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(data, vmax=1e7)
    plt.show()