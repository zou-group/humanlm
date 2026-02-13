import re

from gradio import Request
from gradio.context import Context


def sanitize_filename(value: str) -> str:
    """Replace special characters with a hyphen to make a string safe for file paths."""
    sanitized = re.sub(r'[^\w\-]', '-', value)
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    return sanitized.strip('-')


def persist(component, cookies):
    """Persist a Gradio component's state across page reloads using cookies."""
    sessions = cookies

    def resume_session(value, request: Request):
        return sessions.get(request.cookies.get('user_id', ""), value)

    def update_session(value, request: Request):
        sessions[request.cookies.get('user_id', "")] = value

    Context.root_block.load(resume_session, inputs=[component], outputs=[component])
    component.change(update_session, inputs=[component])

    return component
