// Higher-precision ql::Constants for one xpmath backend.
// Included from kokkosMaths_xp.h inside namespace ql.
// FloatFloat is not specialized: it is coarser than the double tables
// in kokkosMaths.h, and the double-double cutoffs would underflow to zero.
//
// Chebyshev (43) and Bernoulli (25) words come from the quad literals used
// by the ddfun tables. DoubleDouble stores both IEEE doubles. QuadFloat and
// TripleFloat store the same values split into floats.

#pragma once

#if defined(XPMATH_BACKEND_dd)
    template<>
    KOKKOS_INLINE_FUNCTION
    int Constants<xp_real>::_num_C() { return 43; }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_C(int i) {
        const xp_real coeffs[43] = {
            xp_real::from_bits(0x3fdb849409b3171fULL, 0xbc61d08606ea8094ULL), // C[0],
            xp_real::from_bits(0x3fda39817bf08e86ULL, 0xbc708f7bc068d1c3ULL), // C[1],
            xp_real::from_bits(0xbf9308d8ddfc0d4fULL, 0xbbf1bfa97f0c6941ULL), // C[2],
            xp_real::from_bits(0x3f57e13e5937f304ULL, 0xbbf969cc9f4aca3cULL), // C[3],
            xp_real::from_bits(0xbf22bfb0166773adULL, 0xbbbf2b286495c7dbULL), // C[4],
            xp_real::from_bits(0x3ef0a7ded9492ebfULL, 0x3b8490eeb0ccf2dcULL), // C[5],
            xp_real::from_bits(0xbec0011368001ad0ULL, 0x3b5b7a1b649f1629ULL), // C[6],
            xp_real::from_bits(0x3e903cb34eb28bc7ULL, 0xbb361a0a29ecc366ULL), // C[7],
            xp_real::from_bits(0xbe6124e51374ab95ULL, 0xbadc3fca1c01886fULL), // C[8],
            xp_real::from_bits(0x3e32a9df9d2cf366ULL, 0x3ab088ab369bb63bULL), // C[9],
            xp_real::from_bits(0xbe04d08a1a397c33ULL, 0xba735fddea7b4414ULL), // C[10],
            xp_real::from_bits(0x3dd7ac399e1b3a10ULL, 0x3a7ac270c61db429ULL), // C[11],
            xp_real::from_bits(0xbdab5cf6e979608fULL, 0xba4c202e357f882eULL), // C[12],
            xp_real::from_bits(0x3d8008088ec141cbULL, 0x3a24e69a8e63ff1fULL), // C[13],
            xp_real::from_bits(0xbd53011b4100aa8bULL, 0x39f924a9ad769bbcULL), // C[14],
            xp_real::from_bits(0x3d26c16ffed236b0ULL, 0x39c2c37e09229bd6ULL), // C[15],
            xp_real::from_bits(0xbcfb7c91e18c27c2ULL, 0xb978259da7d88ef0ULL), // C[16],
            xp_real::from_bits(0x3cd0ba8c8168432dULL, 0xb97f38b87f87b056ULL), // C[17],
            xp_real::from_bits(0xbca48079ae714341ULL, 0x3938662e035d673fULL), // C[18],
            xp_real::from_bits(0x3c794781af84652eULL, 0xb8df1ac5eacaaed6ULL), // C[19],
            xp_real::from_bits(0xbc4f573ccb1c7256ULL, 0x38c0cea00a8fc228ULL), // C[20],
            xp_real::from_bits(0x3c23861c064989cfULL, 0x38b9858a8264065fULL), // C[21],
            xp_real::from_bits(0xbbf86f1e6a569636ULL, 0x389bb8f28a19ffd1ULL), // C[22],
            xp_real::from_bits(0x3bceb443cb9151fdULL, 0x3865a1ff47ab26b3ULL), // C[23],
            xp_real::from_bits(0xbba35d221a0bf286ULL, 0xb8416f4aacf5c45eULL), // C[24],
            xp_real::from_bits(0x3b7881fe4d78bb1dULL, 0xb815c2b246c97c6fULL), // C[25],
            xp_real::from_bits(0xbb4f1dd5a2fc7960ULL, 0x37cad4f86fb2f160ULL), // C[26],
            xp_real::from_bits(0x3b23cfc3b677dfe1ULL, 0xb7a9e8649f3ef895ULL), // C[27],
            xp_real::from_bits(0xbaf94bd49715a0bdULL, 0x379861cde38fb0f0ULL), // C[28],
            xp_real::from_bits(0x3ad030bf7644b683ULL, 0xb77791ac8f6b139dULL), // C[29],
            xp_real::from_bits(0xbaa4c60ebb342bc8ULL, 0xb74eefcd74311948ULL), // C[30],
            xp_real::from_bits(0x3a7ab68a8997736cULL, 0x370ce9ceb1387136ULL), // C[31],
            xp_real::from_bits(0xba5135f1c3d828d0ULL, 0xb6fa3081db7911a2ULL), // C[32],
            xp_real::from_bits(0x3a2638548f4222a9ULL, 0x36c71521bbeedd19ULL), // C[33],
            xp_real::from_bits(0xb9fcbd5755a6d334ULL, 0xb683ea3a0ae4390aULL), // C[34],
            xp_real::from_bits(0x39d29e317705e440ULL, 0xb66845af574cbaefULL), // C[35],
            xp_real::from_bits(0xb9a82938b74d8e42ULL, 0x3645649ddd2643d2ULL), // C[36],
            xp_real::from_bits(0x397f67066462e3f2ULL, 0xb5f576d5d6bcf559ULL), // C[37],
            xp_real::from_bits(0xb9546fcb719bbdcfULL, 0x35f7299ec8a3fdadULL), // C[38],
            xp_real::from_bits(0x392aa30fd9290536ULL, 0x35ca760528288e40ULL), // C[39],
            xp_real::from_bits(0xb90161be6c7ded9aULL, 0xb573d0e195f44a48ULL), // C[40],
            xp_real::from_bits(0x38d6b6861e77d634ULL, 0x356bfef4bcfab248ULL), // C[41],
            xp_real::from_bits(0xb8adb6cee2774df0ULL, 0x354562db4ea6970aULL), // C[42]
        };
        return coeffs[i];
    }
    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_B(int i) {
        const xp_real coeffs[25] = {
            xp_real::from_bits(0x3f9c71c71c71c71cULL, 0x3c3c71c71c71c71cULL), // B[0],
            xp_real::from_bits(0xbf323456789abcdfULL, 0xbb723456789abcdfULL), // B[1],
            xp_real::from_bits(0x3ed3d079fb6ef3e3ULL, 0x3b4adb9557cf6495ULL), // B[2],
            xp_real::from_bits(0xbe78a86a49f629d1ULL, 0x3b19b054db95c888ULL), // B[3],
            xp_real::from_bits(0x3e204d7f65caf373ULL, 0xbac6cd713a6fb97aULL), // B[4],
            xp_real::from_bits(0xbdc658a4b8f16a75ULL, 0x3a4e68e462783b95ULL), // B[5],
            xp_real::from_bits(0x3d6f63f1e311ac24ULL, 0x39ffbe3b291e07ccULL), // B[6],
            xp_real::from_bits(0xbd16731c59dbd7deULL, 0xb9b968f9b1e5279dULL), // B[7],
            xp_real::from_bits(0x3cc04805fdce7819ULL, 0xb93eaa712fafb00dULL), // B[8],
            xp_real::from_bits(0xbc67e168b15d7793ULL, 0x38eb10f259d71460ULL), // B[9],
            xp_real::from_bits(0x3c11ac70a7618abdULL, 0xb8bf797e5f69a3dbULL), // B[10],
            xp_real::from_bits(0xbbba5bf70e5eefd2ULL, 0xb85e19ea7e4b4c05ULL), // B[11],
            xp_real::from_bits(0x3b63c8881c2dd68cULL, 0x37f33be648f8a92aULL), // B[12],
            xp_real::from_bits(0xbb0ddc14c868f2dbULL, 0xb76503d00b03c1baULL), // B[13],
            xp_real::from_bits(0x3ab6a45025fc86a2ULL, 0x375b9645dce4fc4aULL), // B[14],
            xp_real::from_bits(0xba613d916dfdf3ecULL, 0x37082c02b91f6e2cULL), // B[15],
            xp_real::from_bits(0x3a0a5a26479b86c0ULL, 0xb6a24fe24f17e162ULL), // B[16],
            xp_real::from_bits(0xb9b434a4aab89c3dULL, 0xb64e9ad4ef4e6d6fULL), // B[17],
            xp_real::from_bits(0x395f138b40614ceaULL, 0x35f312f1b88436c6ULL), // B[18],
            xp_real::from_bits(0xb907f5f58356eed3ULL, 0xb5ad1122bddef73cULL), // B[19],
            xp_real::from_bits(0x38b284be7dea0f2dULL, 0xb54a892964d5f581ULL), // B[20],
            xp_real::from_bits(0xb85cafd4d179e626ULL, 0xb4db92ba45b7ea02ULL), // B[21],
            xp_real::from_bits(0x380643613376ba51ULL, 0x3489f1f3b8056581ULL), // B[22],
            xp_real::from_bits(0xb7b14f2ea64ff5d8ULL, 0xb429bd0e711af996ULL), // B[23],
            xp_real::from_bits(0x375af5da8e3360cdULL, 0x33f5ccf27901f02bULL), // B[24]
        };
        return coeffs[i];
    }
    template<>
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_qlonshellcutoff() {
        return xp_real(1e-20);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_eps() { return xp_real(1e-12); }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_neglig() { return xp_real(1e-28); }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_reps() { return xp_real(1e-30); }

#elif defined(XPMATH_BACKEND_qf)
    template<>
    KOKKOS_INLINE_FUNCTION
    int Constants<xp_real>::_num_C() { return 43; }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_C(int i) {
        const xp_real coeffs[43] = {
            xp_real::from_bits(0x3edc24a0U, 0x321b3172U, 0xa491d086U, 0x965d5013U), // C[0],
            xp_real::from_bits(0x3ed1cc0cU, 0xb181ee2fU, 0xa5108f7cU, 0x187e5cb9U), // C[1],
            xp_real::from_bits(0xbc9846c7U, 0x2f00fcacU, 0x227b9016U, 0x95bf8635U), // C[2],
            xp_real::from_bits(0x3abf09f3U, 0xadd901a0U, 0x21669633U, 0x14c16a6cU), // C[3],
            xp_real::from_bits(0xb915fd81U, 0x2c9988c5U, 0x1fb06a6cU, 0x92c92b90U), // C[4],
            xp_real::from_bits(0x37853ef7U, 0xaad6da28U, 0x9d56de23U, 0x10c333cbU), // C[5],
            xp_real::from_bits(0xb600089bU, 0xa98001adU, 0x1adbd0dbU, 0x0d93e2c5U), // C[6],
            xp_real::from_bits(0x3481e59aU, 0x286b28bcU, 0x1bd4f2fbU, 0x8e27b30eU), // C[7],
            xp_real::from_bits(0xb3092729U, 0x26c8b547U, 0x9a21c3fdU, 0x0dbc7fcfU), // C[8],
            xp_real::from_bits(0x31954efdU, 0xa434c326U, 0x97f7bbaaU, 0x8b49644aU), // C[9],
            xp_real::from_bits(0xb0268451U, 0x2338d07aU, 0x96c26bfcU, 0x0a05612fU), // C[10],
            xp_real::from_bits(0x2ebd61cdU, 0xa0f262f8U, 0x13d61386U, 0x06c3b685U), // C[11],
            xp_real::from_bits(0xad5ae7b7U, 0xa0979609U, 0x128f7f47U, 0x05a8077dU), // C[12],
            xp_real::from_bits(0x2c004044U, 0x1fec141dU, 0x93158cb3U, 0x068e63ffU), // C[13],
            xp_real::from_bits(0xaa9808daU, 0x9c805545U, 0x901b6d59U, 0x83944b22U), // C[14],
            xp_real::from_bits(0x29360b80U, 0x9b16e4a8U, 0x0e161bf0U, 0x019229bdU), // C[15],
            xp_real::from_bits(0xa7dbe48fU, 0x99c613e1U, 0x8bc12cedU, 0x800fb11eU), // C[16],
            xp_real::from_bits(0x2685d464U, 0x18b42196U, 0x0c031d1eU, 0x0000f09fU), // C[17],
            xp_real::from_bits(0xa52403cdU, 0x98e71434U, 0x8acf33a4U, 0x00006badU), // C[18],
            xp_real::from_bits(0x23ca3c0dU, 0x17784653U, 0x8a01f1acU, 0x8002f565U), // C[19],
            xp_real::from_bits(0xa27ab9e6U, 0x95b1c725U, 0x893de62cU, 0x000002a4U), // C[20],
            xp_real::from_bits(0x211c30e0U, 0x1449313aU, 0x86ccf4ebU, 0x0000004dU), // C[21],
            xp_real::from_bits(0x9fc378f3U, 0x93256963U, 0x86b22387U, 0x00000451U), // C[22],
            xp_real::from_bits(0x1e75a21eU, 0x11b91520U, 0x84aa5e01U, 0x00000048U), // C[23],
            xp_real::from_bits(0x9d1ae911U, 0x103e81afU, 0x035d216bU, 0x8000000bU), // C[24],
            xp_real::from_bits(0x1bc40ff2U, 0x0f578bb2U, 0x8255c2b2U, 0x80000002U), // C[25],
            xp_real::from_bits(0x9a78eeadU, 0x8cbf1e58U, 0x0006b53eU, 0x00000000U), // C[26],
            xp_real::from_bits(0x191e7e1eU, 0x8c988202U, 0x000e617aU, 0x80000000U), // C[27],
            xp_real::from_bits(0x97ca5ea5U, 0x0b0ea5f4U, 0x0006c30eU, 0x00000000U), // C[28],
            xp_real::from_bits(0x168185fcU, 0x8a1bb498U, 0x000150ddU, 0x80000000U), // C[29],
            xp_real::from_bits(0x95263076U, 0x08197a87U, 0x800007bcU, 0x00000000U), // C[30],
            xp_real::from_bits(0x13d5b454U, 0x07197737U, 0x8000078cU, 0x80000000U), // C[31],
            xp_real::from_bits(0x9289af8eU, 0x85760a34U, 0x80000034U, 0x80000000U), // C[32],
            xp_real::from_bits(0x1131c2a4U, 0x04f4222bU, 0x8000006aU, 0x80000000U), // C[33],
            xp_real::from_bits(0x8fe5eabbU, 0x032592cdU, 0x80000008U, 0x80000000U), // C[34],
            xp_real::from_bits(0x0e94f18cU, 0x820fa1bcU, 0x80000000U, 0x80000000U), // C[35],
            xp_real::from_bits(0x8d4149c6U, 0x008b271cU, 0x80000000U, 0x80000000U), // C[36],
            xp_real::from_bits(0x0bfb3833U, 0x0008c5c8U, 0x80000000U, 0x80000000U), // C[37],
            xp_real::from_bits(0x8aa37e5cU, 0x00073221U, 0x00000000U, 0x00000000U), // C[38],
            xp_real::from_bits(0x0955187fU, 0x80006d70U, 0x00000000U, 0x00000000U), // C[39],
            xp_real::from_bits(0x880b0df3U, 0x800031f8U, 0x00000000U, 0x00000000U), // C[40],
            xp_real::from_bits(0x06b5b431U, 0x800000c4U, 0x80000000U, 0x80000000U), // C[41],
            xp_real::from_bits(0x856db677U, 0x80000027U, 0x80000000U, 0x80000000U), // C[42]
        };
        return coeffs[i];
    }
    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_B(int i) {
        const xp_real coeffs[25] = {
            xp_real::from_bits(0x3ce38e39U, 0xaf638e39U, 0x21e38e39U, 0x94638e39U), // B[0],
            xp_real::from_bits(0xb991a2b4U, 0x2ceca864U, 0x1f7edcbbU, 0x92cf1358U), // B[1],
            xp_real::from_bits(0x369e83d0U, 0xa9922184U, 0x1d435b73U, 0x90aa0c27U), // B[2],
            xp_real::from_bits(0xb3c54352U, 0xa71f629dU, 0x99193eadU, 0x0cdcae44U), // B[3],
            xp_real::from_bits(0x31026bfbU, 0x24395e6eU, 0x17a9328fU, 0x8ae9bee6U), // B[4],
            xp_real::from_bits(0xae32c526U, 0x2161d2b1U, 0x14c79a39U, 0x0744f077U), // B[5],
            xp_real::from_bits(0x2b7b1f8fU, 0x1dc46b09U, 0x0ffdf1d9U, 0x0311e07dU), // B[6],
            xp_real::from_bits(0xa8b398e3U, 0x1bc48504U, 0x0ecd2e0dU, 0x8247949eU), // B[7],
            xp_real::from_bits(0x26024030U, 0x988c61faU, 0x0bf0aac7U, 0x001a0a0aU), // B[8],
            xp_real::from_bits(0xa33f0b46U, 0x16ea2887U, 0x89b93bc3U, 0x8001a629U), // B[9],
            xp_real::from_bits(0x208d6385U, 0x13ec3158U, 0x875f797eU, 0x80000bedU), // B[10],
            xp_real::from_bits(0x9dd2dfb8U, 0x9165eefdU, 0x841e19eaU, 0x8000003fU), // B[11],
            xp_real::from_bits(0x1b1e4441U, 0x8df48a5dU, 0x002677cdU, 0x80000000U), // B[12],
            xp_real::from_bits(0x986ee0a6U, 0x8b868f2eU, 0x0013eafcU, 0x00000000U), // B[13],
            xp_real::from_bits(0x15b52281U, 0x08bf90d4U, 0x00004dcbU, 0x00000000U), // B[14],
            xp_real::from_bits(0x9309ec8bU, 0x86dfdf3fU, 0x00000461U, 0x80000000U), // B[15],
            xp_real::from_bits(0x1052d132U, 0x037370d8U, 0x80000001U, 0x80000000U), // B[16],
            xp_real::from_bits(0x8da1a525U, 0x812b89c4U, 0x00000000U, 0x00000000U), // B[17],
            xp_real::from_bits(0x0af89c5aU, 0x000030a6U, 0x00000000U, 0x00000000U), // B[18],
            xp_real::from_bits(0x883fafacU, 0x80000d5cU, 0x00000000U, 0x00000000U), // B[19],
            xp_real::from_bits(0x059425f4U, 0x80000043U, 0x00000000U, 0x00000000U), // B[20],
            xp_real::from_bits(0x82e57ea7U, 0x00000007U, 0x00000000U, 0x00000000U), // B[21],
            xp_real::from_bits(0x00590d85U, 0x80000000U, 0x80000000U, 0x80000000U), // B[22],
            xp_real::from_bits(0x800229e6U, 0x00000000U, 0x00000000U, 0x00000000U), // B[23],
            xp_real::from_bits(0x00000d7bU, 0x80000000U, 0x80000000U, 0x80000000U), // B[24]
        };
        return coeffs[i];
    }
    template<>
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_qlonshellcutoff() {
        return xp_real(1e-20);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_eps() { return xp_real(1e-12); }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_neglig() { return xp_real(1e-25); }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_reps() { return xp_real(1e-27); }

#elif defined(XPMATH_BACKEND_tf)
    template<>
    KOKKOS_INLINE_FUNCTION
    int Constants<xp_real>::_num_C() { return 43; }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_C(int i) {
        const xp_real coeffs[43] = {
            xp_real::from_bits(0x3edc24a0U, 0x321b3172U, 0xa491d086U), // C[0],
            xp_real::from_bits(0x3ed1cc0cU, 0xb181ee2fU, 0xa5108f7cU), // C[1],
            xp_real::from_bits(0xbc9846c7U, 0x2f00fcacU, 0x227b9016U), // C[2],
            xp_real::from_bits(0x3abf09f3U, 0xadd901a0U, 0x21669633U), // C[3],
            xp_real::from_bits(0xb915fd81U, 0x2c9988c5U, 0x1fb06a6cU), // C[4],
            xp_real::from_bits(0x37853ef7U, 0xaad6da28U, 0x9d56de23U), // C[5],
            xp_real::from_bits(0xb600089bU, 0xa98001adU, 0x1adbd0dbU), // C[6],
            xp_real::from_bits(0x3481e59aU, 0x286b28bcU, 0x1bd4f2fbU), // C[7],
            xp_real::from_bits(0xb3092729U, 0x26c8b547U, 0x9a21c3fdU), // C[8],
            xp_real::from_bits(0x31954efdU, 0xa434c326U, 0x97f7bbaaU), // C[9],
            xp_real::from_bits(0xb0268451U, 0x2338d07aU, 0x96c26bfcU), // C[10],
            xp_real::from_bits(0x2ebd61cdU, 0xa0f262f8U, 0x13d61386U), // C[11],
            xp_real::from_bits(0xad5ae7b7U, 0xa0979609U, 0x128f7f47U), // C[12],
            xp_real::from_bits(0x2c004044U, 0x1fec141dU, 0x93158cb3U), // C[13],
            xp_real::from_bits(0xaa9808daU, 0x9c805545U, 0x901b6d59U), // C[14],
            xp_real::from_bits(0x29360b80U, 0x9b16e4a8U, 0x0e161bf0U), // C[15],
            xp_real::from_bits(0xa7dbe48fU, 0x99c613e1U, 0x8bc12cedU), // C[16],
            xp_real::from_bits(0x2685d464U, 0x18b42196U, 0x0c031d1eU), // C[17],
            xp_real::from_bits(0xa52403cdU, 0x98e71434U, 0x8acf33a4U), // C[18],
            xp_real::from_bits(0x23ca3c0dU, 0x17784653U, 0x8a01f1acU), // C[19],
            xp_real::from_bits(0xa27ab9e6U, 0x95b1c725U, 0x893de62cU), // C[20],
            xp_real::from_bits(0x211c30e0U, 0x1449313aU, 0x86ccf4ebU), // C[21],
            xp_real::from_bits(0x9fc378f3U, 0x93256963U, 0x86b22387U), // C[22],
            xp_real::from_bits(0x1e75a21eU, 0x11b91520U, 0x84aa5e01U), // C[23],
            xp_real::from_bits(0x9d1ae911U, 0x103e81afU, 0x035d216bU), // C[24],
            xp_real::from_bits(0x1bc40ff2U, 0x0f578bb2U, 0x8255c2b2U), // C[25],
            xp_real::from_bits(0x9a78eeadU, 0x8cbf1e58U, 0x0006b53eU), // C[26],
            xp_real::from_bits(0x191e7e1eU, 0x8c988202U, 0x000e617aU), // C[27],
            xp_real::from_bits(0x97ca5ea5U, 0x0b0ea5f4U, 0x0006c30eU), // C[28],
            xp_real::from_bits(0x168185fcU, 0x8a1bb498U, 0x000150ddU), // C[29],
            xp_real::from_bits(0x95263076U, 0x08197a87U, 0x800007bcU), // C[30],
            xp_real::from_bits(0x13d5b454U, 0x07197737U, 0x8000078cU), // C[31],
            xp_real::from_bits(0x9289af8eU, 0x85760a34U, 0x80000034U), // C[32],
            xp_real::from_bits(0x1131c2a4U, 0x04f4222bU, 0x8000006aU), // C[33],
            xp_real::from_bits(0x8fe5eabbU, 0x032592cdU, 0x80000008U), // C[34],
            xp_real::from_bits(0x0e94f18cU, 0x820fa1bcU, 0x80000000U), // C[35],
            xp_real::from_bits(0x8d4149c6U, 0x008b271cU, 0x80000000U), // C[36],
            xp_real::from_bits(0x0bfb3833U, 0x0008c5c8U, 0x80000000U), // C[37],
            xp_real::from_bits(0x8aa37e5cU, 0x00073221U, 0x00000000U), // C[38],
            xp_real::from_bits(0x0955187fU, 0x80006d70U, 0x00000000U), // C[39],
            xp_real::from_bits(0x880b0df3U, 0x800031f8U, 0x00000000U), // C[40],
            xp_real::from_bits(0x06b5b431U, 0x800000c4U, 0x80000000U), // C[41],
            xp_real::from_bits(0x856db677U, 0x80000027U, 0x80000000U), // C[42]
        };
        return coeffs[i];
    }
    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_B(int i) {
        const xp_real coeffs[25] = {
            xp_real::from_bits(0x3ce38e39U, 0xaf638e39U, 0x21e38e39U), // B[0],
            xp_real::from_bits(0xb991a2b4U, 0x2ceca864U, 0x1f7edcbbU), // B[1],
            xp_real::from_bits(0x369e83d0U, 0xa9922184U, 0x1d435b73U), // B[2],
            xp_real::from_bits(0xb3c54352U, 0xa71f629dU, 0x99193eadU), // B[3],
            xp_real::from_bits(0x31026bfbU, 0x24395e6eU, 0x17a9328fU), // B[4],
            xp_real::from_bits(0xae32c526U, 0x2161d2b1U, 0x14c79a39U), // B[5],
            xp_real::from_bits(0x2b7b1f8fU, 0x1dc46b09U, 0x0ffdf1d9U), // B[6],
            xp_real::from_bits(0xa8b398e3U, 0x1bc48504U, 0x0ecd2e0dU), // B[7],
            xp_real::from_bits(0x26024030U, 0x988c61faU, 0x0bf0aac7U), // B[8],
            xp_real::from_bits(0xa33f0b46U, 0x16ea2887U, 0x89b93bc3U), // B[9],
            xp_real::from_bits(0x208d6385U, 0x13ec3158U, 0x875f797eU), // B[10],
            xp_real::from_bits(0x9dd2dfb8U, 0x9165eefdU, 0x841e19eaU), // B[11],
            xp_real::from_bits(0x1b1e4441U, 0x8df48a5dU, 0x002677cdU), // B[12],
            xp_real::from_bits(0x986ee0a6U, 0x8b868f2eU, 0x0013eafcU), // B[13],
            xp_real::from_bits(0x15b52281U, 0x08bf90d4U, 0x00004dcbU), // B[14],
            xp_real::from_bits(0x9309ec8bU, 0x86dfdf3fU, 0x00000461U), // B[15],
            xp_real::from_bits(0x1052d132U, 0x037370d8U, 0x80000001U), // B[16],
            xp_real::from_bits(0x8da1a525U, 0x812b89c4U, 0x00000000U), // B[17],
            xp_real::from_bits(0x0af89c5aU, 0x000030a6U, 0x00000000U), // B[18],
            xp_real::from_bits(0x883fafacU, 0x80000d5cU, 0x00000000U), // B[19],
            xp_real::from_bits(0x059425f4U, 0x80000043U, 0x00000000U), // B[20],
            xp_real::from_bits(0x82e57ea7U, 0x00000007U, 0x00000000U), // B[21],
            xp_real::from_bits(0x00590d85U, 0x80000000U, 0x80000000U), // B[22],
            xp_real::from_bits(0x800229e6U, 0x00000000U, 0x00000000U), // B[23],
            xp_real::from_bits(0x00000d7bU, 0x80000000U, 0x80000000U), // B[24]
        };
        return coeffs[i];
    }
    template<>
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_qlonshellcutoff() {
        return xp_real(1e-20);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_eps() { return xp_real(1e-12); }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_neglig() { return xp_real(1e-18); }

    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_reps() { return xp_real(1e-20); }

#endif
