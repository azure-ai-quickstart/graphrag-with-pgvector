import os
import subprocess

import streamlit as st


def create_session_files():
    run_command(f"mkdir -p /tmp/{st.session_state.id}/input")
    run_command(f"cp -r ./ragtest/prompts /tmp/{st.session_state.id}/prompts")
    run_command(f"cp -r ./ragtest/settings.yaml /tmp/{st.session_state.id}/settings.yaml")


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            s = output.strip()
            if s.startswith('🚀'):
                st.write(s)

    rc = process.poll()
    return rc


def list_subdirectories(path="/app/ragtest/output"):
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    subdirs.sort(reverse=True)
    return subdirs
