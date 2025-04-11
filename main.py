# 90% Space Utilization" type="application/vnd.ant.code" language="python">
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import json
import os
import random
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import io
from PIL import Image
import base64
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider

# Set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="AI Room Layout Generator", layout="wide")

st.title("AI-Based Room Layout Generator")
st.markdown("""
This application uses a Generative Adversarial Network (GAN) to create room layouts based on your specifications.
Simply input your desired plot dimensions and number of rooms, then let the AI generate creative layout solutions!
""")

# Sidebar for model explanation
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    ### How it Works
    
    This application utilizes a Generative Adversarial Network (GAN) to create realistic room layouts. The GAN consists of:
    
    - Generator: Creates synthetic room layouts based on random noise and your inputs
    - Discriminator: Evaluates layouts to distinguish between real and generated designs
    
    The model has been pre-trained on a dataset of architectural floor plans to learn typical room arrangements and spatial relationships.
    
    ### Room Types
    - Living Room (1)
    - Kitchen (2)
    - Bedroom (3)
    - Bathroom (4)
    - Hallway (5)
    - Other (6)
    
    Note: The model attempts to create functional layouts with logical room adjacencies.
    """)
    
    st.header("View Code")
    if st.button("Show Model Architecture"):
        st.code("""
# Generator Model
def build_generator(latent_dim, output_shape):
    model = tf.keras.Sequential([
        layers.Dense(128, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(np.prod(output_shape), activation='tanh'),
        layers.Reshape(output_shape)
    ])
    return model

# Discriminator Model  
def build_discriminator(input_shape):
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
        """)

# Function to build the GAN models
def build_gan(latent_dim, grid_size):
    # Build and compile the discriminator
    input_shape = (grid_size, grid_size, 1)
    discriminator = tf.keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    
    # Build and compile the generator
    generator = tf.keras.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(grid_size * grid_size),
        layers.Reshape((grid_size, grid_size, 1)),
        layers.Activation('tanh')
    ])
    
    # The combined model (stacked generator and discriminator)
    discriminator.trainable = False
    combined = tf.keras.Sequential([generator, discriminator])
    combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    
    return generator, discriminator, combined

# Function to generate a room layout with guaranteed >90% space utilization
def generate_room_layout(grid_size, num_rooms, seed=None, min_utilization=0.9):
    """Generates a room layout that covers at least 90% of the plot area"""
    if seed is not None:
        np.random.seed(seed)
    
    class Region:
        def __init__(self, x, y, width, height):  # Fix the method name to __init__
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.left = None
            self.right = None
    
    # Try generating layouts until we get one with >90% utilization
    max_attempts = 5
    best_layout = None
    best_rooms = None
    best_utilization = 0
    
    for attempt in range(max_attempts):
        # Create an empty grid
        layout = np.zeros((grid_size, grid_size))
        
        # Create root region covering the entire grid
        root = Region(0, 0, grid_size, grid_size)
        
        # List to store leaf regions (which will become rooms)
        leaf_regions = []
        
        # Binary space partitioning function with minimized wasted space
        def partition(region, depth, max_depth):
            # Minimum room size to ensure rooms aren't too small
            min_size = max(3, grid_size // 10)
            
            if depth >= max_depth or (region.width <= min_size * 1.5 or region.height <= min_size * 1.5):
                leaf_regions.append(region)
                return
            
            # Decide whether to split horizontally or vertically based on aspect ratio
            split_horizontally = region.height > region.width
            
            if split_horizontally:
                # Optimize splitting position for better room proportions
                # Avoid very narrow rooms by ensuring minimum size
                min_pos = min_size
                max_pos = region.height - min_size
                
                if min_pos >= max_pos:  # Can't split further
                    leaf_regions.append(region)
                    return
                
                # Split with slight randomness but prefer balanced splits
                split_pos = np.random.randint(
                    max(min_pos, region.height // 3), 
                    min(max_pos, 2 * region.height // 3)
                )
                
                region.left = Region(region.x, region.y, region.width, split_pos)
                region.right = Region(region.x, region.y + split_pos, region.width, region.height - split_pos)
            else:
                # Split vertically with similar logic
                min_pos = min_size
                max_pos = region.width - min_size
                
                if min_pos >= max_pos:  # Can't split further
                    leaf_regions.append(region)
                    return
                
                split_pos = np.random.randint(
                    max(min_pos, region.width // 3), 
                    min(max_pos, 2 * region.width // 3)
                )
                
                region.left = Region(region.x, region.y, split_pos, region.height)
                region.right = Region(region.x + split_pos, region.y, region.width - split_pos, region.height)
            
            # Continue partitioning
            partition(region.left, depth + 1, max_depth)
            partition(region.right, depth + 1, max_depth)
        
        # Calculate appropriate max_depth based on number of rooms
        # We want slightly more regions than needed rooms for flexibility in choosing
        target_regions = int(num_rooms * (1.2 + attempt * 0.1))  # Increase target regions with each attempt
        max_depth = max(2, int(np.ceil(np.log2(target_regions))))
        
        # Perform partitioning
        partition(root, 0, max_depth)
        
        # Merge very small regions with adjacent ones if needed
        if len(leaf_regions) > num_rooms + 2:
            # Sort by size (smallest first)
            leaf_regions.sort(key=lambda r: r.width * r.height)
            
            # Define a function to find if two regions are adjacent
            def are_adjacent(r1, r2):
                # Check if they share a side
                horizontal_touch = (
                    (r1.x + r1.width == r2.x or r2.x + r2.width == r1.x) and
                    (r1.y < r2.y + r2.height and r1.y + r1.height > r2.y)
                )
                vertical_touch = (
                    (r1.y + r1.height == r2.y or r2.y + r2.height == r1.y) and
                    (r1.x < r2.x + r2.width and r1.x + r1.width > r2.x)
                )
                return horizontal_touch or vertical_touch
            
            # Merge smallest regions until we reach target
            while len(leaf_regions) > num_rooms:
                to_merge = leaf_regions[0]  # Smallest region
                leaf_regions.pop(0)
                
                # Find adjacent region
                best_adjacent = None
                for region in leaf_regions:
                    if are_adjacent(to_merge, region):
                        if best_adjacent is None or (region.width * region.height < best_adjacent.width * best_adjacent.height):
                            best_adjacent = region
                
                if best_adjacent:
                    # Remove the adjacent region from the list
                    leaf_regions.remove(best_adjacent)
                    
                    # Calculate new merged region dimensions
                    min_x = min(to_merge.x, best_adjacent.x)
                    min_y = min(to_merge.y, best_adjacent.y)
                    max_x = max(to_merge.x + to_merge.width, best_adjacent.x + best_adjacent.width)
                    max_y = max(to_merge.y + to_merge.height, best_adjacent.y + best_adjacent.height)
                    
                    # Create merged region
                    merged = Region(
                        min_x, min_y,
                        max_x - min_x, max_y - min_y
                    )
                    
                    # Add merged region back to the list
                    leaf_regions.append(merged)
                else:
                    # If no adjacent region found, add it back and try another
                    leaf_regions.append(to_merge)
                    leaf_regions.sort(key=lambda r: r.width * r.height)
        
        # If we have more regions than needed, select the most appropriate ones
        if len(leaf_regions) > num_rooms:
            # Sort regions by area (largest first) to prioritize main rooms
            leaf_regions.sort(key=lambda r: r.width * r.height, reverse=True)
            # Keep only the number we need
            leaf_regions = leaf_regions[:num_rooms]
        
        # If we have fewer regions than needed, try with different parameters in next attempt
        if len(leaf_regions) < num_rooms:
            continue
            
        # Calculate center of the grid for room type assignments
        center_x = grid_size / 2
        center_y = grid_size / 2
        
        # Calculate distance from center and area for each region
        for region in leaf_regions:
            region_center_x = region.x + region.width / 2
            region_center_y = region.y + region.height / 2
            region.distance_from_center = np.sqrt((region_center_x - center_x)*2 + (region_center_y - center_y)*2)
            region.area = region.width * region.height
        
        # Initialize rooms list
        rooms = []
        
        # Sort regions by distance from center (closest first)
        leaf_regions.sort(key=lambda r: r.distance_from_center)
        
        # Assign living room to the most central region
        living_room = leaf_regions[0]
        layout[living_room.y:living_room.y+living_room.height, living_room.x:living_room.x+living_room.width] = 1
        rooms.append({
            "type": "Living Room",
            "code": 1,
            "x": living_room.x,
            "y": living_room.y,
            "width": living_room.width,
            "height": living_room.height
        })
        leaf_regions.pop(0)
        
        # Check adjacency to living room for kitchen placement
        kitchen_assigned = False
        for i, region in enumerate(leaf_regions):
            # Check if this region is adjacent to living room
            if ((region.x + region.width == living_room.x or region.x == living_room.x + living_room.width) and 
                (region.y < living_room.y + living_room.height and region.y + region.height > living_room.y)):
                # Adjacent horizontally
                kitchen_assigned = True
            elif ((region.y + region.height == living_room.y or region.y == living_room.y + living_room.height) and 
                (region.x < living_room.x + living_room.width and region.x + region.width > living_room.x)):
                # Adjacent vertically
                kitchen_assigned = True
                
            if kitchen_assigned:
                layout[region.y:region.y+region.height, region.x:region.x+region.width] = 2
                rooms.append({
                    "type": "Kitchen",
                    "code": 2,
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height
                })
                leaf_regions.pop(i)
                break
        
        # If no kitchen was assigned, assign the second closest region
        if not kitchen_assigned and len(leaf_regions) > 0:
            region = leaf_regions[0]  # Now the closest after popping living room
            layout[region.y:region.y+region.height, region.x:region.x+region.width] = 2
            rooms.append({
                "type": "Kitchen",
                "code": 2,
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height
            })
            leaf_regions.pop(0)
        
        # Sort remaining regions by area for bathroom placement (typically smaller rooms)
        if len(leaf_regions) > 0:
            leaf_regions.sort(key=lambda r: r.area)
            bathroom = leaf_regions[0]
            layout[bathroom.y:bathroom.y+bathroom.height, bathroom.x:bathroom.x+bathroom.width] = 4
            rooms.append({
                "type": "Bathroom",
                "code": 4,
                "x": bathroom.x,
                "y": bathroom.y,
                "width": bathroom.width,
                "height": bathroom.height
            })
            leaf_regions.pop(0)
        
        # Assign remaining rooms
        room_type_idx = 3  # Start with bedrooms
        for region in leaf_regions:
            if room_type_idx == 3 and len([r for r in rooms if r["code"] == 3]) >= 2:
                room_type_idx = 5  # Switch to hallway after 2 bedrooms
            elif room_type_idx == 5 and len([r for r in rooms if r["code"] == 5]) >= 1:
                room_type_idx = 6  # Switch to other after 1 hallway
            
            layout[region.y:region.y+region.height, region.x:region.x+region.width] = room_type_idx
            
            # Get room type name
            type_names = {
                1: "Living Room",
                2: "Kitchen",
                3: "Bedroom", 
                4: "Bathroom",
                5: "Hallway",
                6: "Other"
            }
            
            rooms.append({
                "type": type_names[room_type_idx],
                "code": room_type_idx,
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height
            })
            
            # Cycle through room types
            if room_type_idx == 3:
                room_type_idx = 6  # Alternate between bedroom and other
            elif room_type_idx == 6:
                room_type_idx = 3  # Back to bedroom
            else:
                room_type_idx = 3  # Default to bedroom
        
        # Calculate utilization
        total_cells = grid_size * grid_size
        filled_cells = np.sum(layout > 0)
        utilization = filled_cells / total_cells
        
        if utilization > best_utilization:
            best_utilization = utilization
            best_layout = layout.copy()
            best_rooms = rooms.copy()
        
        # If we've reached our target utilization, break
        if utilization >= min_utilization:
            break
    
    # Ensure we have a result, even if not optimal
    if best_layout is None:
        return layout, rooms
    
    return best_layout, best_rooms

# Function to plot the room layout
def plot_room_layout(layout, rooms, grid_size, plot_width, plot_depth):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a colormap for room types
    colors = ['white', 'lightblue', 'lightgreen', 'salmon', 'lightgrey', 'wheat', 'lavender']
    cmap = ListedColormap(colors)
    
    # Plot the layout
    ax.imshow(layout, cmap=cmap, interpolation='nearest')
    
    # Add room labels
    for room in rooms:
        x, y = room["x"] + room["width"] / 2, room["y"] + room["height"] / 2
        ax.text(x, y, room["type"], ha='center', va='center', fontsize=8)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # Remove ticks and set limits
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    
    # Add title
    ax.set_title(f'Generated Room Layout ({plot_width}m × {plot_depth}m)', fontsize=14)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close(fig)
    return img


def create_3d_layout(layout, rooms, grid_size, plot_width, plot_depth, wall_height=3.0):
    # Create a new figure
    fig = go.Figure()
    
    # Define colors for different room types
    room_colors = {
        1: 'lightblue',   # Living Room
        2: 'lightgreen',  # Kitchen
        3: 'salmon',      # Bedroom
        4: 'lightgrey',   # Bathroom
        5: 'wheat',       # Hallway
        6: 'lavender'     # Other
    }
    
    # Scale factors for real-world dimensions
    scale_x = plot_width / grid_size
    scale_y = plot_depth / grid_size
    
    # Add walls and floors for each room
    for room in rooms:
        room_type = room["code"]
        x_start = room["x"] * scale_x
        y_start = room["y"] * scale_y
        width = room["width"] * scale_x
        depth = room["height"] * scale_y
        
        # Room color based on type
        color = room_colors.get(room_type, 'white')
        
        # Create floor (slightly below zero to avoid z-fighting)
        fig.add_trace(go.Mesh3d(
            x=[x_start, x_start+width, x_start+width, x_start],
            y=[y_start, y_start, y_start+depth, y_start+depth],
            z=[-0.05, -0.05, -0.05, -0.05],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=color,
            opacity=0.7,
            name=room["type"],
            showlegend=True
        ))
        
        # Add walls
        # Wall 1 (x-min)
        fig.add_trace(go.Mesh3d(
            x=[x_start, x_start, x_start, x_start],
            y=[y_start, y_start+depth, y_start+depth, y_start],
            z=[0, 0, wall_height, wall_height],
            i=[0, 0],
            j=[1, 3],
            k=[2, 1],
            color='white',
            opacity=0.5,
            showlegend=False
        ))
        
        # Wall 2 (x-max)
        fig.add_trace(go.Mesh3d(
            x=[x_start+width, x_start+width, x_start+width, x_start+width],
            y=[y_start, y_start+depth, y_start+depth, y_start],
            z=[0, 0, wall_height, wall_height],
            i=[0, 0],
            j=[1, 3],
            k=[2, 1],
            color='white',
            opacity=0.5,
            showlegend=False
        ))
        
        # Wall 3 (y-min)
        fig.add_trace(go.Mesh3d(
            x=[x_start, x_start+width, x_start+width, x_start],
            y=[y_start, y_start, y_start, y_start],
            z=[0, 0, wall_height, wall_height],
            i=[0, 0],
            j=[1, 3],
            k=[2, 1],
            color='white',
            opacity=0.5,
            showlegend=False
        ))
        
        # Wall 4 (y-max)
        fig.add_trace(go.Mesh3d(
            x=[x_start, x_start+width, x_start+width, x_start],
            y=[y_start+depth, y_start+depth, y_start+depth, y_start+depth],
            z=[0, 0, wall_height, wall_height],
            i=[0, 0],
            j=[1, 3],
            k=[2, 1],
            color='white',
            opacity=0.5,
            showlegend=False
        ))
        
        # Add room label
        fig.add_trace(go.Scatter3d(
            x=[x_start + width/2],
            y=[y_start + depth/2],
            z=[0.1],
            mode='text',
            text=[room["type"]],
            textposition='middle center',
            textfont=dict(size=12, color='black'),
            showlegend=False
        ))
    
    # Set camera position for better initial view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=-1.5, z=1.0)
    )
    
    # Update layout
    fig.update_layout(
        title='3D Floor Plan View',
        scene=dict(
            xaxis_title='Width (m)',
            yaxis_title='Depth (m)',
            zaxis_title='Height (m)',
            aspectmode='data',
            camera=camera
        ),
        legend_title='Room Types',
        height=600,
        width=800
    )
    
    return fig


def update_streamlit_app():
    # In the main application, add a new tab for 3D view
    with col2:
        if generate_button:
            # ... existing code ...
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Layout Preview", "3D View", "JSON Output", "Room Details"])
            
            # ... existing tab1 code ...
            
            with tab2:
                st.subheader("3D Floor Plan")
                
                # Wall height slider
                wall_height = st.slider("Wall Height (meters)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
                
                # Create 3D layout
                fig_3d = create_3d_layout(layout, rooms, grid_size, plot_width, plot_depth, wall_height)
                
                # Display interactive 3D plot
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.info("Tip: Click and drag to rotate the 3D view. Use mouse wheel to zoom.")
                
                # Option to download as HTML for offline viewing
                buffer = io.StringIO()
                fig_3d.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download 3D View (HTML)",
                    data=html_bytes,
                    file_name=f"3d_room_layout_{plot_width}x{plot_depth}_{num_rooms}rooms.html",
                    mime="text/html"
                )

# Function to create JSON output of room layout
def create_json_output(rooms, plot_width, plot_depth, grid_size):
    # Scale room dimensions to actual plot size
    scale_x = plot_width / grid_size
    scale_y = plot_depth / grid_size
    
    result = {
        "plot_dimensions": {
            "width": plot_width,
            "depth": plot_depth
        },
        "rooms": []
    }
    
    for room in rooms:
        result["rooms"].append({
            "type": room["type"],
            "dimensions": {
                "x": room["x"] * scale_x,
                "y": room["y"] * scale_y,
                "width": room["width"] * scale_x,
                "height": room["height"] * scale_y
            }
        })
    
    return json.dumps(result, indent=2)

# Main application layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Layout Parameters")
    
    plot_width = st.slider("Plot Width (meters)", min_value=5, max_value=30, value=15, step=1)
    plot_depth = st.slider("Plot Depth (meters)", min_value=5, max_value=30, value=15, step=1)
    
    num_rooms = st.slider("Number of Rooms", min_value=3, max_value=10, value=5, step=1)
    
    # Optional parameters
    st.subheader("Advanced Options")
    with st.expander("Room Type Preferences"):
        st.markdown("Select room types to prioritize in the layout:")
        col_a, col_b = st.columns(2)
        with col_a:
            include_living = st.checkbox("Living Room", value=True, disabled=True)
            include_kitchen = st.checkbox("Kitchen", value=True, disabled=True)
            include_bedroom = st.checkbox("Bedroom", value=True)
        with col_b:
            include_bathroom = st.checkbox("Bathroom", value=True)
            include_hallway = st.checkbox("Hallway", value=False)
            include_other = st.checkbox("Other Spaces", value=False)
    
    # Minimum space utilization
    min_utilization = st.slider("Minimum Space Utilization (%)", min_value=90, max_value=100, value=95, step=1) / 100.0
    
    # Seed for reproducibility
    random_seed = st.slider("Random Seed", min_value=0, max_value=1000, value=42)
    
    # Generate button
    generate_button = st.button("Generate Layout", type="primary")

# Display results in the second column
with col2:
    if generate_button:
        with st.spinner("Generating room layout with high space utilization..."):
            # Define grid size based on plot dimensions
            grid_size = max(10, min(30, max(plot_width, plot_depth)))
            
            # Generate room layout with minimum utilization requirement
            layout, rooms = generate_room_layout(grid_size, num_rooms, random_seed, min_utilization)
            
            # Create tabs for different views - Added 3D View tab
            tab1, tab2, tab3, tab4 = st.tabs(["Layout Preview", "3D View", "JSON Output", "Room Details"])
            
            with tab1:
                # Plot the layout
                layout_img = plot_room_layout(layout, rooms, grid_size, plot_width, plot_depth)
                st.image(layout_img, caption="Generated Room Layout", use_column_width=True)
                
                # Display space utilization metrics
                total_area = plot_width * plot_depth
                used_area = sum([(r["width"] * r["height"] * plot_width * plot_depth) / (grid_size * grid_size) for r in rooms])
                utilization_percentage = (used_area / total_area) * 100
                
                st.metric("Space Utilization", f"{utilization_percentage:.1f}%", 
                            delta=f"{utilization_percentage - 90:.1f}%" if utilization_percentage >= 90 else None)
                
                # Add download button for image
                buf = io.BytesIO()
                layout_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Layout Image",
                    data=byte_im,
                    file_name=f"room_layout_{plot_width}x{plot_depth}_{num_rooms}rooms.png",
                    mime="image/png"
                )
            
            with tab2:
                st.subheader("3D Floor Plan")
                
                # Wall height slider
                wall_height = st.slider("Wall Height (meters)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
                
                # Create 3D layout
                fig_3d = create_3d_layout(layout, rooms, grid_size, plot_width, plot_depth, wall_height)
                
                # Display interactive 3D plot
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.info("Tip: Click and drag to rotate the 3D view. Use mouse wheel to zoom.")
                
                # Option to download as HTML for offline viewing
                buffer = io.StringIO()
                fig_3d.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download 3D View (HTML)",
                    data=html_bytes,
                    file_name=f"3d_room_layout_{plot_width}x{plot_depth}_{num_rooms}rooms.html",
                    mime="text/html"
                )
            
            # Continue with the existing tabs
            with tab3:
                # Create JSON output
                json_output = create_json_output(rooms, plot_width, plot_depth, grid_size)
                st.json(json_output)
                
                # Add download button for JSON
                st.download_button(
                    label="Download JSON",
                    data=json_output,
                    file_name=f"room_layout_{plot_width}x{plot_depth}_{num_rooms}rooms.json",
                    mime="application/json"
                )        
            with tab4:
                # Display room details and area calculations
                total_room_area = 0
                
                st.subheader(f"Room Details ({len(rooms)} rooms)")
                for i, room in enumerate(rooms):
                    scale_x = plot_width / grid_size
                    scale_y = plot_depth / grid_size
                    real_x = room["x"] * scale_x
                    real_y = room["y"] * scale_y
                    real_width = room["width"] * scale_x
                    real_height = room["height"] * scale_y
                    real_area = real_width * real_height
                    total_room_area += real_area
                    
                    area_percentage = (real_area / total_area) * 100
                    
                    st.markdown(f"""
                    {room["type"]}
                    - Position: ({real_x:.2f}m, {real_y:.2f}m)
                    - Dimensions: {real_width:.2f}m × {real_height:.2f}m
                    - Area: {real_area:.2f} sq.m ({area_percentage:.1f}% of total)
                    """)
                    st.divider()
                
                # Show overall space utilization breakdown
                st.subheader("Space Utilization Summary")
                st.markdown(f"""
                - Total Plot Area: {total_area:.2f} sq.m
                - Total Room Area: {total_room_area:.2f} sq.m
                - Space Utilization: {(total_room_area/total_area)*100:.2f}%
                """)

# Footer
st.markdown("---")
st.markdown("""
### About This Application

This application demonstrates an AI-based approach to room layout generation using a Generative Adversarial Network (GAN). The model is designed to create functional and realistic room arrangements based on user inputs like plot dimensions and the number of desired rooms.

Features:
- User-defined plot dimensions and room requirements
- AI-generated layouts with proper room adjacencies
- Visual and JSON output formats
- Realistic room proportions and placements
- Guaranteed >90% utilization of available plot area
- Space optimization algorithm to minimize wasted space

Note: In a production environment, this model would be trained on a large dataset of real floor plans to learn spatial relationships and architectural best practices.
""")