def read_inputs(primary_fluids, secondary_fluids, pcs):
    inputs_accepted = False
    while not inputs_accepted:
        primary_accepted = False
        while not primary_accepted:
            primary_fluid = input("Primary fluid: ")
            if primary_fluid not in primary_fluids:
                print("Your primary fluid selection is not currently supported. Currently supported fluids are:")
                print(*primary_fluids, sep=", ")
                continue
            primary_accepted = True
        secondary_accepted = False
        while not secondary_accepted:
            secondary_fluid = input("Secondary fluid: ")
            if secondary_fluid not in secondary_fluids:
                print("Your secondary fluid selection is not currently supported. Currently supported fluids are:")
                print(*secondary_fluids, sep=", ")
                continue
            secondary_accepted = True
        heat = input("Thermal power (in kW): ")

        final_check = input("You have selected " + primary_fluid + " as your primary fluid, "
            + "coupled to a(n) " + pcs[secondary_fluid] + " with a thermal power of " + heat + "kW."
            + " Are these the inputs you would like to use? (Y/N): ")
        if final_check == "N":
            continue
        inputs_accepted = True
    inputs = {
        "primary_fluid": primary_fluid,
        "secondary_fluid": secondary_fluid,
        "pcs": pcs
    }
    return inputs