"""Google Analytics 4 integration for Streamlit.

Uses st.components.v1.html to inject GA4 into the parent window
(not the iframe), which is the only reliable way to run JS in Streamlit.
"""

import streamlit.components.v1 as components


def inject_ga4(measurement_id: str) -> None:
    """Inject GA4 tracking script that targets the parent Streamlit window."""
    if not measurement_id:
        return

    components.html(
        f"""
        <script>
            // Inject gtag.js into the PARENT window (Streamlit's main page)
            const parent = window.parent.document;

            // Only inject once
            if (!parent.getElementById('ga4-gtag')) {{
                const gtagScript = parent.createElement('script');
                gtagScript.id = 'ga4-gtag';
                gtagScript.async = true;
                gtagScript.src = 'https://www.googletagmanager.com/gtag/js?id={measurement_id}';
                parent.head.appendChild(gtagScript);

                const inlineScript = parent.createElement('script');
                inlineScript.id = 'ga4-config';
                inlineScript.textContent = `
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){{dataLayer.push(arguments);}}
                    gtag('js', new Date());
                    gtag('config', '{measurement_id}', {{
                        send_page_view: true,
                        page_location: window.location.href,
                        page_title: document.title
                    }});
                `;
                parent.head.appendChild(inlineScript);
            }}
        </script>
        """,
        height=0,
        width=0,
    )


def track_event(
    measurement_id: str,
    event_name: str,
    params: dict = None,
) -> None:
    """Fire a custom GA4 event on the parent window."""
    if not measurement_id:
        return

    params = params or {}
    params_js = ", ".join(f"'{k}': '{v}'" for k, v in params.items())

    components.html(
        f"""
        <script>
            if (window.parent && window.parent.gtag) {{
                window.parent.gtag('event', '{event_name}', {{{params_js}}});
            }}
        </script>
        """,
        height=0,
        width=0,
    )
