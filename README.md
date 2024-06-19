<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Huawei Parkinson Disease Analysis & Detection</h3>

  <p align="center">
  This is a Parkinson Disease analysis library with Huawei smart watch devices by detecting the patient's specific activities.
    <br />
    <a href="https://github.com/sparksoftltd/parkingson_disease_analysis_detection"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection">View Demo</a>
    ·
    <a href="https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

PD project description here...

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started


To get a local copy up and running follow these simple example steps.

### Prerequisites

### `PyENV` installation
This is an example of how to list things you need to use the software and how to install them.
* `pyenv` is an exceptionally versatile tool designed to handle multiple Python versions on a single machine. It simplifies the process of switching between various Python versions, making it an ideal choice for projects with specific version needs. This feature is incredibly beneficial in development settings where testing and compatibility with multiple Python versions are critical.

  ```sh
   # Update Linux core dependencies
   sudo apt update

   # Install required libraries for Debian/Ubuntu Linux
   sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

   # Download and execute the PyENV installer script
   curl https://pyenv.run | bash

   # Append these lines to your ~/.bashrc (or ~/.zshrc) file
   echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc

   # Activate the new settings by refreshing your shell
   exec "$SHELL"

   # Update Pyenv to latest version
   pyenv update

   # Ensure that PyENV is correctly installed
   pyenv --version
  ```

### Installation Python and virtual environment

1. Install the appropriate Python version

   ```sh
   # List the available Python versions
   pyenv install --list

   # Install the Python version, default use 3.8.19 for this project compatbility
   pyenv install 3.8.19
   ```

2. Create a virtual env for this project

   ```sh
   pyenv virtualenv 3.8.19 PD_3.8.19
   ```

3. Activate the virtual enviroment above

   ```sh
   pyenv activate PD_3.8.19
   ```

### Install the project required dependencies

1. Under the above virtual env and in the project root directory and run

   ```sh
   pip install -r requirements.txt
   ```



### `Rclone` installation

Rclone is a command-line program to manage files on cloud storage. We use this program to sync our data in `./input` and `./output` folder to/from AWS s3 cloud storage.
</p>

`Beware: if you update the folder you were not supposed to work on, there is a possibility that you may delete or update your peer colleagues' data, make sure if you are updating the data folder you work on`

I have created a few Make command so that you only update the specific folders such as `activity`, `feature/extraction`, `feature/selection`.


1. Install `Rclone`

  To install rclone on Linux/macOS/BSD systems, run:

  ```sh
  sudo -v ; curl https://rclone.org/install.sh | sudo bash
  ```

  To install rclone on Windows, go to below page to download executable. </p>
  https://rclone.org/install/

2. Configure `Rclone`
  You dont need to do anything, no matter which OS you work on, because I have prepare the rclone configuration file in this project, it is located at ./rclone.conf, and it has AWS secret, please use it accordingly without sharing them with others.


3. Practise the data sync between your local PC and remote AWS S3 storage. Please copy your most updated data into the project `input` and `output` folder accordingly and use the commands below in your project directory.

  To sync the activity  input data
  ```sh
  make sync-input-activity-data
  ```

  To sync the activity output data
  ```sh
  make sync-output-activity-data
  ```

  To sync the feature extraction input data
  ```sh
  make sync-input-feature-extraction-data
  ```

  To sync the feature extraction output data
  ```sh
  make sync-output-feature-extraction-data
  ```

  To sync the feature selection input data
  ```sh
  make sync-input-feature-selection-data
  ```

  To sync the feature selectio output data
  ```sh
  make sync-output-feature-selection-data
  ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Repo implementation
- [ ] Repo improvement
    - [ ] Algorithm evaluation
    - [ ] Data augmentation

See the [open issues](https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Peng Yue - [@twitter_handle](https://x.com/pengyue) - peng.yue@sparksoft.uk

Project Link: [https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection](https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/sparksoftltd/huawei-smartwatch-parkingson-disease-detection.svg?style=for-the-badge
[contributors-url]: https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sparksoftltd/huawei-smartwatch-parkingson-disease-detection.svg?style=for-the-badge
[forks-url]: https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection/network/members
[stars-shield]: https://img.shields.io/github/stars/sparksoftltd/huawei-smartwatch-parkingson-disease-detection.svg?style=for-the-badge
[stars-url]: https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection/stargazers
[issues-shield]: https://img.shields.io/github/issues/sparksoftltd/huawei-smartwatch-parkingson-disease-detection.svg?style=for-the-badge
[issues-url]: https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection/issues
[license-shield]: https://img.shields.io/github/license/gsparksoftltd/huawei-smartwatch-parkingson-disease-detection.svg?style=for-the-badge
[license-url]: https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/peng-yue-a917814/
[product-screenshot]: images/screenshot.png
[Python-url]: https://python.org/
