import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: {
    unoptimized: true
  },
  allowedDevOrigins: ["192.168.1.149"]
};

export default nextConfig;
