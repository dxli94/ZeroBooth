import streamlit as st

from multipage import MultiPage

import generation as gen
import stylization as styl


if __name__ == "__main__":
    app = MultiPage()

    app.add_page("Generation", gen.app)
    app.add_page("Stylization", styl.app)
    app.run()
