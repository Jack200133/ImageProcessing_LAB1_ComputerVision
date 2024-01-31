from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def from_image_to_vectors(image: Image, channel_count=3):
    image_array = np.array(image) 

    if channel_count == 3 and image.mode != 'RGB':
        image = image.convert('RGB')
    elif channel_count == 4 and image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    vectorized_image = image_array.reshape(-1, channel_count)
    return vectorized_image

def initialize_centroids(points, k):
    indices = np.random.permutation(points.shape[0])[:k]
    return points[indices]

def closest_centroid(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(points, closest, k):
    new_centroids = np.array([points[closest == ki].mean(axis=0) for ki in range(k)])
    return new_centroids

def kmeans(points, k, max_iters=100):
    if k<2:
        k=2
 
    centroids = initialize_centroids(points, k)
    for i in range(max_iters):
        closest = closest_centroid(points, centroids)
        new_centroids = update_centroids(points, closest, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        if i==max_iters-1:
            print("Reached max iters")

    return closest, centroids

def build_map_segments(centroids, labels):
    n_centroids = len(centroids)
    
    fig, ax = plt.subplots(figsize=(10, 1))
    
    for i, color in enumerate(centroids):
        ax.barh(y=0, width=1, left=i, height=1, color=color/255, edgecolor='none')
        ax.text(i + 0.5, 0.5, str(i), ha='center', va='center', color='white' if np.mean(color) < 128 else 'black')
    
    ax.set_xlim(0, n_centroids)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.show()

def build_quantized_image(centroids, labels, original_image):
    quantized_image = np.array([centroids[label] for label in labels])
    quantized_image = quantized_image.reshape(original_image.size[1], original_image.size[0], -1).astype(np.uint8)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(quantized_image)
    plt.axis('off')
    plt.title('Quantized Image')
    plt.show()

def build_map_segments_and_quantized_image(centroids, labels, image):
    build_map_segments(centroids, labels)
    build_quantized_image(centroids, labels, image)

def main():
    image_path = "./imgs/plantas_gotas.jpg"
    image = Image.open(image_path) 
    vectorized_image = from_image_to_vectors(image)
    labels, centroids = kmeans(vectorized_image, 7)
    build_map_segments_and_quantized_image(centroids, labels, image)


if __name__ == "__main__":
    main()