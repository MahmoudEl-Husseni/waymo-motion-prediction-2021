while ! ping -c 1 google.com; do
    # Wait for 5 seconds
    sleep 5

    # Try again
done
docker run --gpus all -v "/home/sensai/agent motion prediction":/main cnn_motion_impl
