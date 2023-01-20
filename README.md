# MITHEx
Heat exchanger sizing and cost modeler for Gen IV micro-reactor and SMR technologies. PCS efficiency at given secondary conditions additionally computed.

Usage:

For steam generator analysis, the hot and cold temperatures of each fluid must be included. For non-steam generator analysis, two parameters between mass flow rate, cold temperature, and hot temperature must be included with the third calue left blank. In the case that all three parameters are specified, the mass flow rate will be overwritten. Be careful to specify a mass flow rate which will result in the hot fluid maintaining a higher temperature than the cold fluid. For this reason, it is better to specify temperatures.

The HX length search bounds are important inputs to keep track of. In steam generator modeling, setting too high of an upper bound will result in errors being thrown. In the case that the upper bound is set too high, the iteration parameter will be approximately 0.98-0.99. Try increasing the upper bound slightly if you see this value, and if there is no change, decrease the upper bound.

Be sure to check the pressure of your operating fluid, especially for sCO2, as the tool does not enforce that CO2 is in the supercritical regime.  This is especially important for the cycle efficiency calculations, as both the high and low pressures should be in the supercritical regime.
