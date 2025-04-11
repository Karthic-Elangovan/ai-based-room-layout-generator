# AI Room Layout Generator

## Description
The AI Room Layout Generator is a powerful Streamlit application that creates intelligent room layouts based on user specifications. Utilizing a space optimization algorithm, the application generates functional floor plans that maximize space utilization (guaranteed >90%) while creating logical room arrangements.

## Features
- Generate room layouts based on custom plot dimensions
- Specify the number of rooms needed
- Advanced options for room type preferences
- Minimum space utilization settings
- Interactive 2D and 3D visualization
- Downloadable outputs (PNG, JSON, HTML)
- Room details and area calculations

## Requirements
- Python 3.7+
- Streamlit
- NumPy
- Matplotlib
- TensorFlow
- Plotly
- PIL
- io
- json
- random

## Installation

```bash
# Clone the repository
git clone https://github.com/Karthic-Elangovan/ai-based-room-layout-generator.git
cd ai-based-room-layout-generator

```
``` bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage
1. Set your desired plot dimensions (width and depth in meters)
2. Specify the number of rooms required
3. Adjust advanced options if needed:
   - Room type preferences
   - Minimum space utilization
   - Random seed for reproducibility
4. Click "Generate Layout" to create your floor plan
5. Explore the results in different views:
   - Layout Preview: 2D visualization of the room arrangement
   - 3D View: Interactive 3D model with adjustable wall height
   - JSON Output: Structured data format for further processing
   - Room Details: Specific measurements and utilization statistics

## How It Works
The application uses a binary space partitioning approach to divide the plot into regions that become rooms. The algorithm:

1. Creates an initial grid based on plot dimensions
2. Recursively divides the space using intelligent splitting rules
3. Ensures minimum room sizes for practicality
4. Assigns room types based on spatial relationships (e.g., kitchen adjacent to living room)
5. Optimizes space utilization to minimize wasted areas
6. Generates visual representations of the layout

## Customization
- **Plot Dimensions**: Adjust the width and depth of your plot (5-30 meters)
- **Number of Rooms**: Choose between 3-10 rooms
- **Room Types**: Select which room types to prioritize
- **Space Utilization**: Set minimum space utilization percentage (90-100%)
- **Random Seed**: Control randomness for reproducible results

## Output Formats
- **2D Layout Image**: Visual representation of the floor plan
- **3D Interactive Model**: Explore the space in three dimensions
- **JSON Data**: Structured data for integration with other tools
- **Room Details**: Precise measurements and statistics

## Example Output
The generated layouts include:
- Living Room (always included)
- Kitchen (always included)
- Bedrooms
- Bathroom(s)
- Optional: Hallways and other spaces

Each room includes position coordinates, dimensions, and area calculations.

## Limitations
- The current version focuses on rectangular rooms
- Room adjacency is optimized but may not account for all architectural best practices
- Doors and windows are not explicitly modeled

## Future Enhancements
- Support for non-rectangular room shapes
- Door and window placement
- Furniture layout suggestions
- Style preferences (open plan, traditional, etc.)
- Export to CAD formats

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
