# Visual Vinyl

An interactive musical instrument inspired by synesthesia, vinyl records, and tangible interfaces.

![Visual Vinyl](./IMG_6513.jpg)

## Overview

Visual Vinyl is a tangible music-making system that transforms color and physical arrangement into sound. Users compose music by placing colored magnetic tessellations on a rotating platter. A camera observes the platter, interprets visual information using computer vision, and translates those patterns into musical notes through generative audio synthesis.

The project explores the relationship between sight and sound, drawing inspiration from synesthesia and the physicality of vinyl records.

## Features

* Color-to-note mapping
* Rotating platter sequencer
* Tangible interaction through magnetic pieces
* Real-time computer vision pipeline
* Generative sound synthesis
* Pentatonic scale for harmonious compositions
* Modular acrylic construction
* Custom physical and digital prototyping

## How It Works

Visual Vinyl transforms space into time.

As the platter rotates, a camera continuously observes a small region of the surface. Colors detected within this region are mapped to musical notes and sent to a sound engine.

```text
Colored Tiles
      ↓
Rotating Platter
      ↓
Camera
      ↓
OpenCV
      ↓
Python
      ↓
OSC Messages
      ↓
SuperCollider
      ↓
Audio
```

## System Architecture

### Physical System

* Acrylic enclosure
* Motorized platter
* Lazy Susan bearing
* Magnetic tessellated tiles
* Camera module
* Raspberry Pi
* Speaker

### Software Stack

#### Computer Vision

* Python
* OpenCV

#### Communication

* OSC (Open Sound Control)

#### Sound Synthesis

* SuperCollider

#### Hardware Control

* ESP8266
* DC motor

## Interaction Design

Visual Vinyl explores music creation through physical manipulation rather than traditional screen-based interfaces.

Users compose by:

1. Selecting colored magnetic tiles.
2. Arranging them on the platter.
3. Adjusting rotational speed.
4. Listening to evolving musical patterns.
5. Rearranging pieces to create new compositions.

The result lies somewhere between a turntable, a sequencer, and a kinetic sculpture.

## Inspiration

Visual Vinyl is influenced by:

* Synesthesia
* Vinyl records
* Tangible User Interfaces
* Computational creativity
* Mathematical tessellations
* Generative music
* Physical computing

## Technologies

### Hardware

* Raspberry Pi
* ESP8266
* DC Motor
* Acrylic fabrication
* Magnetic sheets
* Camera module

### Software

* Python
* OpenCV
* SuperCollider
* OSC
* NumPy

## Repository Structure

```text
visual_vinyl/
│
├── color_to_sc.py
├── frame_mask.py
├── tessellation_to_sc.py
├── motor_control/
├── supercollider/
├── assets/
├── prototypes/
├── docs/
└── README.md
```

## Future Directions

* Shape recognition using aperiodic tessellations
* Browser-based version with webcam support
* Multiple instrument modes
* MIDI export
* Recording and looping
* Adaptive scales and tunings
* Collaborative composition

## Authors

Abhinav Mehrotra

University of Colorado Boulder

## License

MIT License
