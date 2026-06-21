import streamlit as st

def get_severity_color(severity_str):
    """
    Returns hex color for severity.
    """
    colors = {
        "LOW": "#2ECC71",       # Emerald Green
        "MODERATE": "#F1C40F",  # Sunflower Yellow
        "HIGH": "#E67E22",      # Carrot Orange
        "CRITICAL": "#E74C3C"   # Alizarin Red
    }
    return colors.get(severity_str.upper(), "#FAFAFA")

def render_metric_card(title: str, value_html: str, badge_html: str, content_html: str, border_color: str):
    """
    Generates a generic beautiful CSS metric card container.
    """
    # De-indented to prevent Markdown code block formatting
    card_html = f"""<div style="background-color: #1A1D26; border-left: 5px solid {border_color}; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
<div style="font-size: 0.85rem; color: #8A8F98; text-transform: uppercase; letter-spacing: 1px; font-weight: bold; margin-bottom: 0.5rem;">{title}</div>
<div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.8rem;">
<span style="font-size: 1.8rem; font-weight: bold; color: #FAFAFA;">{value_html}</span>
{badge_html}
</div>
<div style="font-size: 0.9rem; color: #CFD2D6; line-height: 1.4;">{content_html}</div>
</div>"""
    return card_html

def render_eis_card(eis: float, severity: str):
    """
    Renders Card 1 - Event Impact Score
    """
    color = get_severity_color(severity)
    badge = f"""<span style="background-color: {color}22; color: {color}; border: 1px solid {color}; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">{severity} IMPACT</span>"""
    
    # Operational descriptions based on severity
    if severity == "LOW":
        description = "Traffic flow is normal or slightly slowed down. Normal beats and standard police patrol are sufficient."
    elif severity == "MODERATE":
        description = "Moderate congestion expected. Queue build-up on corridors. Active monitoring and traffic regulation required."
    elif severity == "HIGH":
        description = "High congestion with significant delays. Gridlock risk on intersections. Immediate officer deployment recommended."
    else:
        description = "Critical bottleneck - GRIDLOCK imminent. Major delays on adjacent corridors. Prioritize immediate resources and diversion plans."
        
    return render_metric_card(
        title="Event Impact Score (EIS)",
        value_html=f"{eis:.1f} <span style='font-size: 1rem; color: #8A8F98;'>/ 100</span>",
        badge_html=badge,
        content_html=description,
        border_color=color
    )

def render_manpower_card(manpower: int, is_peak_hour: bool, hour: int):
    """
    Renders Card 2 - Manpower Deployment
    """
    # Peak shift calculation
    if 6 <= hour < 14:
        shift = "Morning Peak"
    elif 14 <= hour < 22:
        shift = "Evening Peak"
    else:
        shift = "Night Shift"
        
    content = f"""<div style="font-weight: 500; color: #3498DB; margin-bottom: 0.3rem;">Priority Shift: {shift}</div>
<div style="color: #8A8F98; font-size: 0.85rem;">Deploy {manpower} personnel. Recommended dispatch based on temporal congestion demand.</div>"""
    
    return render_metric_card(
        title="Officers to Deploy",
        value_html=f"{manpower} <span style='font-size: 1rem; color: #8A8F98;'>officers</span>",
        badge_html="",
        content_html=content,
        border_color="#3498DB"
    )

def render_barricades_card(barricades: int):
    """
    Renders Card 3 - Barricades Required
    """
    pct = min(100.0, (barricades / 50.0) * 100.0)
    
    progress_bar = f"""<div style="background-color: #2C303E; border-radius: 10px; height: 8px; width: 100%; margin-top: 0.5rem; margin-bottom: 0.8rem; overflow: hidden;"><div style="background-color: #9B59B6; height: 100%; width: {pct}%; border-radius: 10px;"></div></div>"""
    
    alert_text = ""
    if barricades > 20:
        alert_text = f"<div style='color: #E67E22; font-weight: bold; font-size: 0.85rem; margin-top: 0.3rem;'>Notice: Request additional barricade truck dispatch</div>"
        
    content = f"""{progress_bar}
<div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #8A8F98;">
<span>0</span>
<span>25 units</span>
<span>50 (Max)</span>
</div>
{alert_text}"""
    
    return render_metric_card(
        title="Barricades Dispatched",
        value_html=f"{barricades} <span style='font-size: 1rem; color: #8A8F98;'>units</span>",
        badge_html="",
        content_html=content,
        border_color="#9B59B6"
    )

def render_diversion_card(diversion: bool):
    """
    Renders Card 4 - Diversion Plan
    """
    if diversion:
        badge = """<span style="background-color: #E74C3C22; color: #E74C3C; border: 1px solid #E74C3C; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">REQUIRED</span>"""
        content = """<div style="background-color: #E74C3C15; color: #E74C3C; border: 1px solid #E74C3C33; padding: 0.5rem 0.8rem; border-radius: 4px; font-weight: 500; font-size: 0.85rem;">DIVERSION REQUIRED: Activate alternate route protocol</div>"""
        border = "#E74C3C"
        val = "YES"
    else:
        badge = """<span style="background-color: #2ECC7122; color: #2ECC71; border: 1px solid #2ECC71; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">NOT REQUIRED</span>"""
        content = """<div style="background-color: #2ECC7115; color: #2ECC71; border: 1px solid #2ECC7133; padding: 0.5rem 0.8rem; border-radius: 4px; font-weight: 500; font-size: 0.85rem;">No diversion required: Local traffic management sufficient</div>"""
        border = "#2ECC71"
        val = "NO"
        
    return render_metric_card(
        title="Diversion Required",
        value_html=val,
        badge_html=badge,
        content_html=content,
        border_color=border
    )
