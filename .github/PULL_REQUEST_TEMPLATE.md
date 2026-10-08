name: Pull request
description: Propose a focused change
body:
  - type: textarea
    id: change
    attributes:
      label: What changed
      description: One concern per PR; list the files and why each one moved
    validations:
      required: true
  - type: textarea
    id: verify
    attributes:
      label: How you verified it
      description: Commands run and their results (make test, make lint, leak tests)
      placeholder: |
        make test — ...
        make lint — ...
    validations:
      required: true
  - type: checkboxes
    id: rules
    attributes:
      label: Checklist
      options:
        - label: Touched only what the change requires
          required: true
        - label: No data committed; no Enron content in code, tests, or screenshots
          required: true
        - label: New tests fail without the change (or no tests needed — say why)
          required: true
