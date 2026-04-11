from Haptic_Device_Setup.haply_barebones import HapticVisualizer

viz = HapticVisualizer(display_plot=False)
if viz.running:
    print("haptic device is running")

if viz.active:
    print("haptic is active")

if viz.a_pressed:
    print("button was pressed")
