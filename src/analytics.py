"""Google Analytics 4 integration for Streamlit.

Injects GA4 directly into the page HTML (not an iframe) so Google
can detect the tag and events fire on the main window.
"""

import streamlit as st


def inject_ga4(measurement_id: str) -> None:
    """Inject Google Analytics 4 tracking into the Streamlit page head."""
    if not measurement_id:
        return

    st.markdown(
        f"""
        <!-- Google Analytics 4 -->
        <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{measurement_id}', {{
                send_page_view: true
            }});
        </script>
        """,
        unsafe_allow_html=True,
    )


def track_event(
    measurement_id: str,
    event_name: str,
    params: dict = None,
) -> None:
    """Send a custom GA4 event via inline script."""
    if not measurement_id:
        return

    params = params or {}
    params_js = ", ".join(f"'{k}': '{v}'" for k, v in params.items())

    st.markdown(
        f"""
        <script>
            if (typeof gtag !== 'undefined') {{
                gtag('event', '{event_name}', {{{params_js}}});
            }}
        </script>
        """,
        unsafe_allow_html=True,
    )
