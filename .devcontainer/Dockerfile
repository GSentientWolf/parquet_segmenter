# Debian Trixie base with Python 3.13, Temurin JDK 17 and developer tools
FROM debian:trixie

ARG DEBIAN_FRONTEND=noninteractive

# Keep image small by cleaning apt caches
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    curl \
    gnupg \
    lsb-release \
    sudo \
    git \
    zsh \
    locales \
  build-essential \
  python3.13 \
  python3.13-venv \
  python3-pip \
  rustc \
  cargo \
  ripgrep \
  fd-find \
  fzf \
  tree \
  procps \
    ca-certificates \
    unzip \
  && rm -rf /var/lib/apt/lists/*

# Ensure fd is available as `fd`
RUN if [ -x "$(command -v fdfind)" ] && [ ! -f /usr/local/bin/fd ]; then ln -s "$(command -v fdfind)" /usr/local/bin/fd || true; fi

# Add Adoptium apt repository and install temurin-17-jdk (fallback to openjdk-17-jdk)
RUN set -eux; \
  apt-get update; \
  wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /usr/share/keyrings/adoptium.gpg; \
  echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(lsb_release -cs) main" > /etc/apt/sources.list.d/adoptium.list || true; \
  apt-get update || true; \
  if apt-get install -y --no-install-recommends temurin-17-jdk; then \
    true; \
  else \
    apt-get install -y --no-install-recommends openjdk-17-jdk || true; \
  fi; \
  rm -rf /var/lib/apt/lists/* || true

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/* || true

# Create vscode user
RUN useradd -m -s /usr/bin/zsh -G sudo vscode \
 && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99_vscode

# Install oh-my-zsh and some plugins for autocomplete
USER vscode
WORKDIR /home/vscode
ENV HOME=/home/vscode
RUN git clone --depth=1 https://github.com/ohmyzsh/ohmyzsh.git $HOME/.oh-my-zsh || true \
 && cp $HOME/.oh-my-zsh/templates/zshrc.zsh-template $HOME/.zshrc || true \
 && git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions $HOME/.oh-my-zsh/custom/plugins/zsh-autosuggestions || true \
 && git clone --depth=1 https://github.com/zsh-users/zsh-completions $HOME/.oh-my-zsh/custom/plugins/zsh-completions || true \
 && sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-completions)/' $HOME/.zshrc || true

# Ensure Python3.13 pip is available
USER root
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python3.13 /tmp/get-pip.py || true \
 && rm -f /tmp/get-pip.py \
 && python3.13 -m pip install --upgrade pip setuptools wheel || true

# Create workspace dir
RUN mkdir -p /workspace && chown -R vscode:vscode /workspace

# Set JAVA_HOME for OpenJDK 17
ENV JAVA_HOME=/usr/lib/jvm
ENV PATH="$JAVA_HOME/bin:${PATH}"

# Default user
USER vscode

CMD ["/usr/bin/zsh"]

# Install dust (Rust crate) into the vscode user's cargo bin
RUN if [ -x "$(command -v cargo)" ]; then mkdir -p $HOME/.cargo/bin && cargo install dust || true; fi
