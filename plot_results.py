import matplotlib.pyplot as plt

def read_log(log_file="train_dqn.log"):
    episodes = []
    rewards = []
    wait_times = []
    queue_lengths = []

    with open(log_file, "r") as file:
        for line in file:
            if "Total Reward" in line:
                parts = line.split(", ")
                ep = int(parts[0].split(" ")[1])
                rew = float(parts[1].split("=")[1])
                wait = float(parts[2].split("=")[1])
                queue = float(parts[3].split("=")[1])

                episodes.append(ep)
                rewards.append(rew)
                wait_times.append(wait)
                queue_lengths.append(queue)

    return episodes, rewards, wait_times, queue_lengths

# Read log file
episodes, rewards, wait_times, queue_lengths = read_log()

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(episodes, rewards, label="Total Reward", color="blue")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total Reward Over Episodes")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(episodes, wait_times, label="Avg Wait Time", color="red")
plt.xlabel("Episode")
plt.ylabel("Avg Wait Time (seconds)")
plt.title("Average Wait Time Over Episodes")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(episodes, queue_lengths, label="Avg Queue Length", color="green")
plt.xlabel("Episode")
plt.ylabel("Avg Queue Length (vehicles)")
plt.title("Queue Length Over Episodes")
plt.legend()

plt.tight_layout()
plt.show()