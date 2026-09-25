"""Short help for the carer: how to choose the words and the photos."""

from nicegui import ui

TITLE: str = "How to add an object"
STEPS: tuple[str, ...] = (
    'Give the object a name, for example "Blue bottle".',
    (
        'Write the kind of object in plain words, for example "a bottle". '
        'Do not write "my favourite bottle". The computer uses these words '
        "to find the object in a photo."
    ),
    (
        "Add 5 to 10 photos. Show only this object. Take the photos from "
        "different sides and in different places."
    ),
    (
        "Check the crop of each photo. Accept it if it shows the object. "
        "Remove the photo if it does not."
    ),
    (
        "The object is ready when 3 photos are accepted. More photos give "
        "a better result."
    ),
)


def enroll_guide() -> None:
    """Build the help. It is closed by default."""
    with (
        ui.expansion(TITLE).classes("up-guide"),
        ui.element("ol").classes("up-guide__steps"),
    ):
        for step in STEPS:
            with ui.element("li").classes("up-guide__step"):
                ui.label(step)
