import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    angle = np.random.random()
    y0 = np.random.random()
    steps = np.random.random(100) * 0.1
    X = np.cumsum(steps)
    Y = angle * X + y0 + np.random.random(100)
    return X, Y


def plot_data(X, Y, W, B):
    plt.scatter(X, Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Generated Data")
    plt.plot(X, W * X + B, color="red")
    plt.show()
    plt.close()


def linear_regression(X, Y, max_steps=1000, lr=0.01):
    W, B = np.random.random(size=2)
    prev_loss = np.inf
    for step in range(max_steps):
        pred = W * X + B
        loss = pred - Y
        loss_delta = prev_loss - np.mean(loss**2)
        prev_loss = np.mean(loss**2)
        if loss_delta < 1e-5:
            print(f"Converged after {step} steps")
            break
        W = W - np.mean(loss * X) * lr
        B = B - np.mean(loss) * lr

    return W, B


def main():
    X, Y = generate_data()
    W, B = linear_regression(X, Y)
    print(f"Learned parameters: W = {W}, B = {B}")
    plot_data(X, Y, W, B)


if __name__ == "__main__":
    main()
