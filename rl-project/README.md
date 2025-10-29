# Reinforcement Learning Project

This project implements a custom reinforcement learning environment using the Gymnasium library. The environment simulates a robot that interacts with a hand, and it is designed to be used with reinforcement learning algorithms from the Stable Baselines3 library.

## Project Structure

```
rl-project
├── src
│   ├── custom_env.py      # Defines the CustomEnv class for the reinforcement learning environment
│   ├── renderer.py        # Contains rendering functionality for visualizing the environment
│   └── main.py            # Entry point for running the application
├── requirements.txt        # Lists the project dependencies
└── README.md               # Documentation for the project
```

## Installation

To set up the project, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Usage

1. **Run the Main Application**: 
   To start the reinforcement learning process, execute the `main.py` file. This will initialize the environment and begin training the model.

   ```
   python src/main.py
   ```

2. **Rendering**: 
   The rendering functionality is handled in the `renderer.py` file. It takes coordinates as input to visualize the robot, hand, and trajectory in the environment.

## Dependencies

The project requires the following Python packages:

- `gymnasium`: For creating the custom reinforcement learning environment.
- `pygame`: For rendering the environment visually.
- `stable-baselines3`: For implementing reinforcement learning algorithms.

## Contributing

Contributions to the project are welcome. If you have suggestions for improvements or new features, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.