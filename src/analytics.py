"""Google Analytics 4 integration for Streamlit.

Injects the GA4 gtag.js script and provides event tracking helpers.
"""

import streamlit as st
import streamlit.components.v1 as components


def inject_ga4(measurement_id: str) -> None:
    """Inject Google Analytics 4 tracking script into the Streamlit app.

    Must be called once at the top of the app, before any other content.
    """
    if not measurement_id:
        return

    ga_script = f"""
        <!-- Google Analytics 4 -->
        <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{measurement_id}', {{
                'send_page_view': true,
                'cookie_flags': 'SameSite=None;Secure'
            }});

            // Custom dimensions for Streamlit
            gtag('set', 'user_properties', {{
                'app_name': 'garmin_virtual_coach',
                'app_version': '1.0.0'
            }});
        </script>
    """
    components.html(ga_script, height=0, width=0)


def track_event(
    measurement_id: str,
    event_name: str,
    params: dict = None,
) -> None:
    """Send a custom GA4 event.

    Args:
        measurement_id: GA4 measurement ID (G-XXXXXXXXXX).
        event_name: Event name (e.g., 'garmin_login', 'coach_chat').
        params: Optional dict of event parameters.
    """
    if not measurement_id:
        return

    params = params or {}
    params_js = ", ".join(f"'{k}': '{v}'" for k, v in params.items())

    event_script = f"""
        <script>
            if (typeof gtag !== 'undefined') {{
                gtag('event', '{event_name}', {{{params_js}}});
            }}
        </script>
    """
    components.html(event_script, height=0, width=0)
