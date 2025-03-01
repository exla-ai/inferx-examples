from exla.models.mobilenet import mobilenet

def main():
    # Create model - it will automatically detect the hardware
    model = mobilenet()

    
    # Example inference with a single image
    single_image = "examples/mobilenet/sample_data/val/image.jpg"
    result = model.inference(single_image)

    print(f"Single image prediction: {result}")
    
    batch_results = model.inference(image1, image2, image3)
    print(f"Batch predictions: {batch_results}")

if __name__ == "__main__":
    main()