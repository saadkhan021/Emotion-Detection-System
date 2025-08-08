# draw_hydroponics_net.py

import graphviz

def build_hydroponics_semantic_net():
    dot = graphviz.Digraph(comment='Hydroponics Semantic Net', format='png')

    # ——— Nutrition System ———
    dot.node('NutritionSystem', 'NutritionSystem', shape='box')
    dot.node('NutrientSolution', 'NutrientSolution', shape='box')
    dot.node('MacroNutrients', 'MacroNutrients', shape='ellipse')
    dot.node('MicroNutrients', 'MicroNutrients', shape='ellipse')
    dot.edge('NutritionSystem', 'NutrientSolution', label='manages')
    dot.edge('NutrientSolution', 'MacroNutrients', label='contains')
    dot.edge('NutrientSolution', 'MicroNutrients', label='contains')

    # ——— Condition Control ———
    dot.node('ConditionControl', 'ConditionControl', shape='box')
    dot.node('Temperature', 'Temperature', shape='ellipse')
    dot.node('pH', 'pH', shape='ellipse')
    dot.node('EC', 'EC', shape='ellipse')
    dot.node('Lighting', 'Lighting', shape='ellipse')
    dot.edge('ConditionControl', 'Temperature', label='regulates')
    dot.edge('ConditionControl', 'pH', label='regulates')
    dot.edge('ConditionControl', 'EC', label='regulates')
    dot.edge('ConditionControl', 'Lighting', label='regulates')

    # ——— Crops and Vegetables ———
    dot.node('Crops', 'Crops', shape='box')
    dot.node('Vegetable', 'Vegetable', shape='ellipse')
    dot.edge('Crops', 'Vegetable', label='includes')

    # Vegetable types
    dot.node('Lettuce', 'Lettuce', shape='box')
    dot.node('Tomato', 'Tomato', shape='box')
    dot.node('Cucumber', 'Cucumber', shape='box')
    dot.edge('Vegetable', 'Lettuce', label='hasType')
    dot.edge('Vegetable', 'Tomato', label='hasType')
    dot.edge('Vegetable', 'Cucumber', label='hasType')

    # ——— Lettuce Stages and Optimal Ranges ———
    dot.node('Lettuce_Seedling', 'Lettuce_Seedling', shape='ellipse')
    dot.node('Lettuce_Vegetative', 'Lettuce_Vegetative', shape='ellipse')
    dot.node('Lettuce_Harvest', 'Lettuce_Harvest', shape='ellipse')
    dot.edge('Lettuce', 'Lettuce_Seedling', label='hasStage')
    dot.edge('Lettuce', 'Lettuce_Vegetative', label='hasStage')
    dot.edge('Lettuce', 'Lettuce_Harvest', label='hasStage')

    # Optimal values for Lettuce Seedling
    dot.node('LS_Temp', '18‒22 °C', shape='plaintext')
    dot.node('LS_pH', '5.5‒6.0', shape='plaintext')
    dot.node('LS_EC', '0.8‒1.2 mS/cm', shape='plaintext')
    dot.node('LS_Light', '100‒150 µmol/m²·s', shape='plaintext')
    dot.edge('Lettuce_Seedling', 'LS_Temp', label='optTemp')
    dot.edge('Lettuce_Seedling', 'LS_pH', label='optpH')
    dot.edge('Lettuce_Seedling', 'LS_EC', label='optEC')
    dot.edge('Lettuce_Seedling', 'LS_Light', label='optLight')

    # Optimal values for Lettuce Vegetative
    dot.node('LV_Temp', '20‒24 °C', shape='plaintext')
    dot.node('LV_pH', '5.8‒6.2', shape='plaintext')
    dot.node('LV_EC', '1.2‒1.6 mS/cm', shape='plaintext')
    dot.node('LV_Light', '200‒250 µmol/m²·s', shape='plaintext')
    dot.edge('Lettuce_Vegetative', 'LV_Temp', label='optTemp')
    dot.edge('Lettuce_Vegetative', 'LV_pH', label='optpH')
    dot.edge('Lettuce_Vegetative', 'LV_EC', label='optEC')
    dot.edge('Lettuce_Vegetative', 'LV_Light', label='optLight')

    # Optimal values for Lettuce Harvest
    dot.node('LH_Temp', '18‒20 °C', shape='plaintext')
    dot.node('LH_pH', '5.8‒6.2', shape='plaintext')
    dot.node('LH_EC', '1.5‒1.8 mS/cm', shape='plaintext')
    dot.node('LH_Light', '150‒200 µmol/m²·s', shape='plaintext')
    dot.edge('Lettuce_Harvest', 'LH_Temp', label='optTemp')
    dot.edge('Lettuce_Harvest', 'LH_pH', label='optpH')
    dot.edge('Lettuce_Harvest', 'LH_EC', label='optEC')
    dot.edge('Lettuce_Harvest', 'LH_Light', label='optLight')

    # Link each parameter back to all Lettuce stages with "optFor"
    for param in ['Temperature', 'pH', 'EC', 'Lighting']:
        dot.edge(param, 'Lettuce_Seedling',    label='optFor')
        dot.edge(param, 'Lettuce_Vegetative',  label='optFor')
        dot.edge(param, 'Lettuce_Harvest',     label='optFor')

    return dot

if __name__ == '__main__':
    g = build_hydroponics_semantic_net()
    # This will produce "hydroponics_semantic_net.png" in the current directory
    output_path = g.render(filename='hydroponics_semantic_net')
    print(f"Semantic net generated: {output_path}")
