#!/bin/sh

PI_SD_CARD_DEVICE=/dev/mmcblk0
PI_SD_CARD_DEVICE_P1=/dev/mmcblk0p1
PI_SD_CARD_DEVICE_P2=/dev/mmcblk0p2

echo 'TODO read this again: https://archlinuxarm.org/platforms/armv8/broadcom/raspberry-pi-4'
ecit 1

sudo fdisk $PI_SD_CARD_DEVICE <<EOF

EOF


echo 'Also add this & a symlink after we partition + mount root: https://ask.fedoraproject.org/t/how-to-connect-to-wifi-on-startup/8083/8'




cat <<EOF
=== Nifty Setup Commands ===

sudo pacman -Syu networkmanager
sudo systemctl enable --now NetworkManager

cat > /wifi_up.sh <<EOD
#!/bin/sh

nmcli device wifi connect 'YOUR_SSID_HERE'
nmcli connection up 'YOUR_SSID_HERE'

EOD

cat > /etc/systemd/system/our-wifi.service <<EOD
[Unit]
Description=auto connect wifi on startup
After=multi-user.target

[Service]
ExecStart=/bin/sh -c /wifi_up.sh

[Install]
WantedBy=multi-user.target

EOD

sudo systemctl enable --now our-wifi.service

# For video stuff
sudo pacman -S opencv python-opencv hdf5


EOF


