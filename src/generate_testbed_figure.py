import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

def draw_grid(ax, title, grid_size, grid_data, arrows=[], lava_annotation=None):
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold', family='serif')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')

    # 1. Draw Grid Lines
    for x in range(grid_size + 1):
        ax.axvline(x, color='#D3D3D3', lw=0.8)
    for y in range(grid_size + 1):
        ax.axhline(y, color='#D3D3D3', lw=0.8)

    # 2. Render Cells (Background Colors + Geometric Shapes)
    for (x, y, bg_color, shape_type, label_text, val_text) in grid_data:
        # Background
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='#E0E0E0', facecolor=bg_color)
        ax.add_patch(rect)
        
        # Center coordinates
        cx, cy = x + 0.5, y + 0.5

        # Geometric Shapes (The Professional Replacement for Emojis)
        if shape_type == 'agent':
            # Blue Triangle
            triangle = patches.RegularPolygon((cx, cy), numVertices=3, radius=0.3, orientation=0, color='#1565C0')
            ax.add_patch(triangle)
        
        elif shape_type == 'goal':
            # Green Star
            ax.plot(cx, cy, marker='*', markersize=20, color='#2E7D32', markeredgecolor='white', markeredgewidth=1)
            
        elif shape_type == 'candy':
            # Gold Diamond
            ax.plot(cx, cy, marker='D', markersize=12, color='#F9A825', markeredgecolor='white', markeredgewidth=1)
            
        elif shape_type == 'lava':
            # Red 'X'
            ax.plot(cx, cy, marker='x', markersize=10, color='#C62828', markeredgewidth=2)

        # Labels (Text)
        if label_text:
            ax.text(cx, y + 0.8, label_text, ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='#333333', family='serif')
            
        # Values (Reward numbers)
        if val_text:
            ax.text(cx, y + 0.15, val_text, ha='center', va='center', 
                    fontsize=12, color='#444444', family='serif')

    # 3. Draw Trajectory Arrows
    for (start, end, style, color, label) in arrows:
        ax.annotate("", xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=2.5, color=color, ls=style))
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=9, 
                    color=color, fontweight='bold', family='serif',
                    bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.9))

    # 4. Annotations (Lava Zone)
    if lava_annotation:
        txt, lx, ly, lcol = lava_annotation
        ax.text(lx, ly, txt, ha='center', va='center', fontsize=10, fontweight='bold', color=lcol, family='serif',
                bbox=dict(facecolor='white', edgecolor='#D3D3D3', boxstyle='round,pad=0.3', alpha=0.9))

def main():
    # --- Professional Palette ---
    c_start, c_goal = '#F5F5F5', '#E8F5E9'  # Very subtle gray/green
    c_lava  = '#FFEBEE' # Very subtle red
    c_candy = '#FFFDE7' # Very subtle yellow
    c_white = '#FFFFFF'
    
    grid_size = 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- SHARED DATA ---
    lava_locs = [(4,4), (4,5), (5,4), (5,5), (3,6), (6,3)]
    # Format: (x, y, bg_color, shape_type, label, value)
    lava_cells = [(x, y, c_lava, 'lava', '', '') for x, y in lava_locs]
    
    lava_note = ("Lava Zone\n(-50)", 4.5, 3.4, '#C62828')

    # --- PANEL A: LATENT TRUTH ---
    data_a = [
        (0, 0, c_start, 'agent', "Start", ""),
        (9, 9, c_goal,  'goal',  "Goal", "+20"),
        (2, 7, c_white, 'candy', "Candy", "-0.1"), 
    ] + lava_cells
    
    arrows_a = [
        ((0.5, 0.5), (0.5, 8.5), '-', '#2E7D32', ""),
        ((0.5, 8.5), (8.5, 8.5), '-', '#2E7D32', "Robust Path"),
        ((8.5, 8.5), (9.0, 9.0), '-', '#2E7D32', "")
    ]

    draw_grid(ax1, r"(a) Latent Objective ($R^*$)", grid_size, data_a, arrows_a, lava_note)

    # --- PANEL B: SOCIAL PROXY ---
    data_b = [
        (0, 0, c_start, 'agent', "Start", ""),
        (9, 9, c_goal,  'goal',  "Goal", "+20"),
        (2, 7, c_candy, 'candy', "Candy", "+10 (Fake)"),
    ] + lava_cells

    arrows_b = [
        ((0.5, 0.5), (2.5, 6.5), '--', '#D32F2F', "Sycophantic Detour"),
    ]
    
    draw_grid(ax2, r"(b) Social Proxy ($R_{soc}$)", grid_size, data_b, arrows_b, lava_note)

    # plt.suptitle("Testbed 1: The Sycophant Trap", fontsize=16, fontweight='bold', family='serif')
    plt.tight_layout()
    
    # Save as PDF (Vector) and PNG (Raster)
    plt.savefig('testbed1_academic.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('testbed1_academic.png', format='png', dpi=300, bbox_inches='tight')
    print("Saved 'testbed1_academic.pdf' and 'testbed1_academic.png'")
    plt.show()

if __name__ == "__main__":
    main()